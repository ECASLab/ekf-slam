#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "ekf_hls.h"  // MAX_N_STATE, HJ_ACTIVE_COLS, MAX_*_SIZE, firma del kernel

int main(int argc, char **argv) {
  const std::string sigma_file = (argc > 1) ? argv[1] : "ekf_sigma.txt";
  const std::string hj_file    = (argc > 2) ? argv[2] : "ekf_hj.txt";
  const std::string xclbin     = (argc > 3) ? argv[3] : "ekf_spmm.xclbin";

  // ── 1. Cargar las matrices reales volcadas por slam ─────────────────────
  arma::mat sigma, Hj;
  if (!sigma.load(sigma_file, arma::arma_ascii)) {
    std::cerr << "ERROR: no pude leer " << sigma_file << "\n";
    return 1;
  }
  if (!Hj.load(hj_file, arma::arma_ascii)) {
    std::cerr << "ERROR: no pude leer " << hj_file << "\n";
    return 1;
  }

  const int n_state = static_cast<int>(sigma.n_rows);
  std::cout << "Cargado: sigma " << sigma.n_rows << "x" << sigma.n_cols
            << " | Hj " << Hj.n_rows << "x" << Hj.n_cols
            << " | n_state=" << n_state << "\n";

  // ── 2. VALIDACIÓN (la barrera anti-cuelgue) ─────────────────────────────
  bool ok = true;
  if (sigma.n_rows != sigma.n_cols) {
    std::cerr << "FALLA: sigma no es cuadrada\n"; ok = false;
  }
  if (static_cast<int>(Hj.n_cols) != n_state || Hj.n_rows != 2) {
    std::cerr << "FALLA: Hj debe ser 2 x n_state\n"; ok = false;
  }
  if (n_state > MAX_N_STATE) {
    std::cerr << "FALLA: n_state (" << n_state << ") > MAX_N_STATE ("
              << MAX_N_STATE << ")\n"; ok = false;
  }

  // Reconstruyo el Hj disperso desde el denso:
  //   columnas 0,1,2 siempre activas (pose) + las 2 columnas del landmark.
  // Las del landmark son las únicas columnas >= 3 con valor no nulo.
  std::vector<int> lm;
  for (int c = 3; c < n_state; ++c) {
    if (Hj(0, c) != 0.0 || Hj(1, c) != 0.0) lm.push_back(c);
  }
  if (lm.size() != 2) {
    std::cerr << "FALLA: esperaba 2 columnas de landmark activas, encontré "
              << lm.size() << ". NO lanzo el kernel.\n";
    ok = false;
  }

  if (!ok) {
    std::cerr << "\nValidación FALLÓ -> aborto ANTES de tocar la FPGA.\n";
    return 2;
  }

  int idx[HJ_ACTIVE_COLS] = {0, 1, 2, lm[0], lm[1]};
  for (int m = 0; m < HJ_ACTIVE_COLS; ++m) {
    if (idx[m] < 0 || idx[m] >= n_state) {
      std::cerr << "FALLA: índice fuera de rango (" << idx[m] << "). Aborto.\n";
      return 2;
    }
  }
  std::cout << "Validación OK. Columnas activas del Hj: {"
            << idx[0] << "," << idx[1] << "," << idx[2] << ","
            << idx[3] << "," << idx[4] << "}\n";

  // ── 3. Abrir device, cargar xclbin, reservar buffers ────────────────────
  auto device = xrt::device(0);
  auto uuid   = device.load_xclbin(xclbin);
  auto kernel = xrt::kernel(device, uuid, "ekf_spmm");

  auto bo_pht     = xrt::bo(device, MAX_PHT_SIZE   * sizeof(float), kernel.group_id(0));
  auto bo_sigma0  = xrt::bo(device, MAX_SIGMA_SIZE * sizeof(float), kernel.group_id(1));
  auto bo_sigma1  = xrt::bo(device, MAX_SIGMA_SIZE * sizeof(float), kernel.group_id(2));
  auto bo_sigma2  = xrt::bo(device, MAX_SIGMA_SIZE * sizeof(float), kernel.group_id(3));
  auto bo_sigma3  = xrt::bo(device, MAX_SIGMA_SIZE * sizeof(float), kernel.group_id(4));
  auto bo_sigma4  = xrt::bo(device, MAX_SIGMA_SIZE * sizeof(float), kernel.group_id(5));
  auto bo_hj_vals = xrt::bo(device, HJ_VALS_SIZE   * sizeof(float), kernel.group_id(6));
  auto bo_hj_idx  = xrt::bo(device, HJ_IDX_SIZE    * sizeof(int),   kernel.group_id(7));

  float *p_pht = bo_pht.map<float *>();
  float *p_s0  = bo_sigma0.map<float *>();
  float *p_s1  = bo_sigma1.map<float *>();
  float *p_s2  = bo_sigma2.map<float *>();
  float *p_s3  = bo_sigma3.map<float *>();
  float *p_s4  = bo_sigma4.map<float *>();
  float *p_hjv = bo_hj_vals.map<float *>();
  int   *p_hji = bo_hj_idx.map<int *>();

  // ── 4. Llenar buffers (double -> float; sigma en row-major) ─────────────
  std::memset(p_s0, 0, MAX_SIGMA_SIZE * sizeof(float));
  for (int i = 0; i < n_state; ++i)
    for (int k = 0; k < n_state; ++k)
      p_s0[i * n_state + k] = static_cast<float>(sigma(i, k));

  const std::size_t nbytes =
      static_cast<std::size_t>(n_state) * n_state * sizeof(float);
  std::memcpy(p_s1, p_s0, nbytes);
  std::memcpy(p_s2, p_s0, nbytes);
  std::memcpy(p_s3, p_s0, nbytes);
  std::memcpy(p_s4, p_s0, nbytes);

  for (int m = 0; m < HJ_ACTIVE_COLS; ++m) {
    p_hji[m]                      = idx[m];
    p_hjv[0 * HJ_ACTIVE_COLS + m] = static_cast<float>(Hj(0, idx[m]));
    p_hjv[1 * HJ_ACTIVE_COLS + m] = static_cast<float>(Hj(1, idx[m]));
  }

  bo_sigma0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_sigma1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_sigma2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_sigma3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_sigma4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_hj_vals.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_hj_idx.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ── 5. Lanzar el kernel UNA vez, con timeout de seguridad ───────────────
  std::cout << "\nLanzando kernel ekf_spmm...\n";
  auto run = kernel(bo_pht, bo_sigma0, bo_sigma1, bo_sigma2, bo_sigma3, bo_sigma4,
                    bo_hj_vals, bo_hj_idx, n_state);

  // Si tu versión de XRT no acepta el timeout con chrono, reemplazá esta línea
  // por:  run.wait();
  ert_cmd_state st = run.wait(std::chrono::milliseconds(2000));
  if (st != ERT_CMD_STATE_COMPLETED) {
    std::cerr << "El kernel NO terminó (estado=" << st << "). "
              << "Recuperá la PL con: sudo xmutil unloadapp; "
              << "sudo xmutil loadapp ekf_spmm\n";
    return 3;
  }

  bo_pht.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // ── 6. Referencia CPU y comparación ─────────────────────────────────────
  arma::mat pht_fpga(n_state, 2);
  for (int i = 0; i < n_state; ++i) {
    pht_fpga(i, 0) = static_cast<double>(p_pht[i * 2 + 0]);
    pht_fpga(i, 1) = static_cast<double>(p_pht[i * 2 + 1]);
  }

  arma::mat pht_ref = sigma * Hj.t();  // n_state x 2
  const double rel = arma::norm(pht_fpga - pht_ref, "fro") /
                     (arma::norm(pht_ref, "fro") + 1e-12);

  std::cout << "\n   fila        FPGA                    CPU\n";
  for (int i = 0; i < std::min(n_state, 6); ++i) {
    std::cout << "   [" << i << "]  ("
              << pht_fpga(i, 0) << ", " << pht_fpga(i, 1) << ")   ("
              << pht_ref(i, 0)  << ", " << pht_ref(i, 1)  << ")\n";
  }

  std::cout << "\nError relativo (Frobenius) = " << rel << "\n";
  const bool pass = (rel < 1e-3);
  std::cout << (pass ? "*** PASS: los datos de slam llegan a la FPGA y vuelven bien ***"
                     : "*** FAIL: revisar el camino de datos ***")
            << "\n";
  return pass ? 0 : 4;
}
