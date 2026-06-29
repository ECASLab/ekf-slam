#ifndef EKF_ACCEL_HPP_
#define EKF_ACCEL_HPP_


#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "ekf_hls.h"

class EkfSpmmAccel
{
public:
  explicit EkfSpmmAccel(const std::string &xclbin_path)
  {
    try {
      device_ = xrt::device(0);
      uuid_   = device_.load_xclbin(xclbin_path);
      kernel_ = xrt::kernel(device_, uuid_, "ekf_spmm");

      const auto g1 = kernel_.group_id(1);
      shared_sigma_ = (kernel_.group_id(2) == g1) && (kernel_.group_id(3) == g1) &&
                      (kernel_.group_id(4) == g1) && (kernel_.group_id(5) == g1);

      bo_pht_     = xrt::bo(device_, MAX_PHT_SIZE   * sizeof(float), kernel_.group_id(0));
      bo_sigma0_  = xrt::bo(device_, MAX_SIGMA_SIZE * sizeof(float), kernel_.group_id(1));
      bo_hj_vals_ = xrt::bo(device_, HJ_VALS_SIZE   * sizeof(float), kernel_.group_id(6));
      bo_hj_idx_  = xrt::bo(device_, HJ_IDX_SIZE    * sizeof(int),   kernel_.group_id(7));

      p_pht_ = bo_pht_.map<float *>();
      p_s0_  = bo_sigma0_.map<float *>();
      p_hjv_ = bo_hj_vals_.map<float *>();
      p_hji_ = bo_hj_idx_.map<int *>();

      if (shared_sigma_) {
        p_s1_ = p_s2_ = p_s3_ = p_s4_ = p_s0_;
      } else {
        bo_sigma1_ = xrt::bo(device_, MAX_SIGMA_SIZE * sizeof(float), kernel_.group_id(2));
        bo_sigma2_ = xrt::bo(device_, MAX_SIGMA_SIZE * sizeof(float), kernel_.group_id(3));
        bo_sigma3_ = xrt::bo(device_, MAX_SIGMA_SIZE * sizeof(float), kernel_.group_id(4));
        bo_sigma4_ = xrt::bo(device_, MAX_SIGMA_SIZE * sizeof(float), kernel_.group_id(5));
        p_s1_ = bo_sigma1_.map<float *>();
        p_s2_ = bo_sigma2_.map<float *>();
        p_s3_ = bo_sigma3_.map<float *>();
        p_s4_ = bo_sigma4_.map<float *>();
      }

      std::memset(p_s0_, 0, MAX_SIGMA_SIZE * sizeof(float));
      if (!shared_sigma_) {
        std::memset(p_s1_, 0, MAX_SIGMA_SIZE * sizeof(float));
        std::memset(p_s2_, 0, MAX_SIGMA_SIZE * sizeof(float));
        std::memset(p_s3_, 0, MAX_SIGMA_SIZE * sizeof(float));
        std::memset(p_s4_, 0, MAX_SIGMA_SIZE * sizeof(float));
      }

      hw_available_ = true;
      std::cerr << "[EkfSpmmAccel] FPGA lista (xclbin=" << xclbin_path
                << ", sigma compartido=" << (shared_sigma_ ? "si" : "no") << ")\n";
    } catch (const std::exception &e) {
      hw_available_ = false;
      std::cerr << "[EkfSpmmAccel] No pude abrir la FPGA (" << e.what()
                << "). Corriendo en CPU.\n";
    }
  }

  ~EkfSpmmAccel()
  {
    std::cerr << "[EkfSpmmAccel] TOTAL  ->  FPGA=" << fpga_count_
              << "   CPU=" << cpu_count_ << "\n";
  }

  arma::mat compute_pht(const arma::mat &sigma, const arma::mat &Hj)
  {
    const auto t0 = std::chrono::steady_clock::now();
    arma::mat r = compute_pht_impl(sigma, Hj);
    const double us = us_since(t0);
    ++op_n_;
    op_sum_   += us;
    op_sumsq_ += us * us;
    if (op_n_ == 1 || us < op_min_) op_min_ = us;
    if (op_n_ == 1 || us > op_max_) op_max_ = us;
    return r;
  }

  void write_summary(const std::string &mode, const std::string &path,
                     double th, double x, double y,
                     double sth, double sx, double sy,
                     double l0x, double l0y)
  {
    const double avg = op_n_ ? op_sum_ / static_cast<double>(op_n_) : 0.0;
    double var = op_n_ ? (op_sumsq_ / static_cast<double>(op_n_) - avg * avg) : 0.0;
    if (var < 0.0) var = 0.0;
    const double sd     = std::sqrt(var);
    const double a_to   = fb_n_ ? sto_sum_   / static_cast<double>(fb_n_) : 0.0;
    const double a_k    = fb_n_ ? sk_sum_    / static_cast<double>(fb_n_) : 0.0;
    const double a_from = fb_n_ ? sfrom_sum_ / static_cast<double>(fb_n_) : 0.0;

    std::ofstream f(path);
    if (!f) {
      std::cerr << "[EkfSpmmAccel] no pude escribir " << path << "\n";
      return;
    }
    f << "mode,total_iterations,avg_op_us,min_op_us,max_op_us,std_op_us,"
         "avg_sync_to_us,avg_kernel_us,avg_sync_from_us,"
         "final_theta,final_x,final_y,final_sigma_theta,final_sigma_x,"
         "final_sigma_y,final_lm0_x,final_lm0_y\n";
    f << mode << "," << op_n_ << "," << avg << "," << op_min_ << "," << op_max_
      << "," << sd << "," << a_to << "," << a_k << "," << a_from << ","
      << th << "," << x << "," << y << "," << sth << "," << sx << "," << sy
      << "," << l0x << "," << l0y << "\n";
    std::cerr << "[EkfSpmmAccel] resumen escrito en " << path
              << " (iteraciones=" << op_n_ << ")\n";
  }

  bool      last_used_fpga()    const { return last_used_fpga_; }
  bool      hw_available()      const { return hw_available_; }
  long long fpga_calls()        const { return fpga_count_; }
  long long cpu_calls()         const { return cpu_count_; }
  double    last_sync_to_us()   const { return last_sync_to_us_; }
  double    last_kernel_us()    const { return last_kernel_us_; }
  double    last_sync_from_us() const { return last_sync_from_us_; }

private:
  static double us_since(const std::chrono::steady_clock::time_point &t0)
  {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
  }

  arma::mat fallback_cpu(const arma::mat &sigma, const arma::mat &Hj)
  {
    ++cpu_count_;
    if (hw_available_ && !announced_cpu_) {
      std::cerr << "[EkfSpmmAccel] cayendo a CPU (la validación rechazó este update)\n";
      announced_cpu_ = true;
    }
    return sigma * Hj.t();
  }

  arma::mat compute_pht_impl(const arma::mat &sigma, const arma::mat &Hj)
  {
    last_used_fpga_ = false;
    const int n_state = static_cast<int>(sigma.n_rows);

    if (!hw_available_)                                            return fallback_cpu(sigma, Hj);
    if (sigma.n_cols != sigma.n_rows)                             return fallback_cpu(sigma, Hj);
    if (static_cast<int>(Hj.n_cols) != n_state || Hj.n_rows != 2) return fallback_cpu(sigma, Hj);
    if (n_state > MAX_N_STATE)                                    return fallback_cpu(sigma, Hj);

    std::vector<int> lm;
    for (int c = 3; c < n_state; ++c)
      if (Hj(0, c) != 0.0 || Hj(1, c) != 0.0) lm.push_back(c);
    if (lm.size() != 2) return fallback_cpu(sigma, Hj);

    const int idx[HJ_ACTIVE_COLS] = {0, 1, 2, lm[0], lm[1]};
    for (int m = 0; m < HJ_ACTIVE_COLS; ++m)
      if (idx[m] < 0 || idx[m] >= n_state) return fallback_cpu(sigma, Hj);

    float *p_s[HJ_ACTIVE_COLS] = {p_s0_, p_s1_, p_s2_, p_s3_, p_s4_};
    for (int m = 0; m < HJ_ACTIVE_COLS; ++m) {
      const int c = idx[m];
      const double *col = sigma.colptr(c);
      float *buf = p_s[m];
      for (int i = 0; i < n_state; ++i)
        buf[i * n_state + c] = static_cast<float>(col[i]);
    }

    for (int m = 0; m < HJ_ACTIVE_COLS; ++m) {
      p_hji_[m]                      = idx[m];
      p_hjv_[0 * HJ_ACTIVE_COLS + m] = static_cast<float>(Hj(0, idx[m]));
      p_hjv_[1 * HJ_ACTIVE_COLS + m] = static_cast<float>(Hj(1, idx[m]));
    }

    // ── sync TO device (cronometrado) ──
    const auto t_a = std::chrono::steady_clock::now();
    bo_sigma0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (!shared_sigma_) {
      bo_sigma1_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_sigma2_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_sigma3_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_sigma4_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    bo_hj_vals_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_hj_idx_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const double t_sync_to = us_since(t_a);

    // ── kernel (cronometrado) ──
    const auto t_b = std::chrono::steady_clock::now();
    try {
      auto launch = [&]() {
        if (shared_sigma_)
          return kernel_(bo_pht_, bo_sigma0_, bo_sigma0_, bo_sigma0_, bo_sigma0_,
                         bo_sigma0_, bo_hj_vals_, bo_hj_idx_, n_state);
        return kernel_(bo_pht_, bo_sigma0_, bo_sigma1_, bo_sigma2_, bo_sigma3_,
                       bo_sigma4_, bo_hj_vals_, bo_hj_idx_, n_state);
      };
      auto run = launch();
      ert_cmd_state st = run.wait(std::chrono::milliseconds(1000));
      if (st != ERT_CMD_STATE_COMPLETED) {
        hw_available_ = false;
        std::cerr << "[EkfSpmmAccel] El kernel no respondió (estado=" << st
                  << "). Deshabilito FPGA y sigo en CPU.\n";
        return fallback_cpu(sigma, Hj);
      }
    } catch (const std::exception &e) {
      hw_available_ = false;
      std::cerr << "[EkfSpmmAccel] Error lanzando el kernel (" << e.what()
                << "). Deshabilito FPGA y sigo en CPU.\n";
      return fallback_cpu(sigma, Hj);
    }
    const double t_kernel = us_since(t_b);

    // ── sync FROM device (cronometrado) ──
    const auto t_c = std::chrono::steady_clock::now();
    bo_pht_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const double t_sync_from = us_since(t_c);

    arma::mat pht(n_state, 2);
    for (int i = 0; i < n_state; ++i) {
      pht(i, 0) = static_cast<double>(p_pht_[i * 2 + 0]);
      pht(i, 1) = static_cast<double>(p_pht_[i * 2 + 1]);
    }

    last_sync_to_us_   = t_sync_to;
    last_kernel_us_    = t_kernel;
    last_sync_from_us_ = t_sync_from;
    ++fb_n_;
    sto_sum_   += t_sync_to;
    sk_sum_    += t_kernel;
    sfrom_sum_ += t_sync_from;

    last_used_fpga_ = true;
    ++fpga_count_;
    if (!announced_fpga_) {
      std::cerr << "[EkfSpmmAccel] primer cálculo en FPGA OK (usando hardware)\n";
      announced_fpga_ = true;
    }
    return pht;
  }

  bool hw_available_   = false;
  bool last_used_fpga_ = false;
  bool shared_sigma_   = false;
  bool announced_fpga_ = false;
  bool announced_cpu_  = false;
  long long fpga_count_ = 0;
  long long cpu_count_  = 0;

  long long op_n_ = 0;
  double op_sum_ = 0.0, op_sumsq_ = 0.0, op_min_ = 0.0, op_max_ = 0.0;
  long long fb_n_ = 0;
  double sto_sum_ = 0.0, sk_sum_ = 0.0, sfrom_sum_ = 0.0;
  double last_sync_to_us_ = 0.0, last_kernel_us_ = 0.0, last_sync_from_us_ = 0.0;

  xrt::device device_;
  xrt::uuid   uuid_;
  xrt::kernel kernel_;

  xrt::bo bo_pht_;
  xrt::bo bo_sigma0_, bo_sigma1_, bo_sigma2_, bo_sigma3_, bo_sigma4_;
  xrt::bo bo_hj_vals_, bo_hj_idx_;

  float *p_pht_ = nullptr;
  float *p_s0_  = nullptr;
  float *p_s1_  = nullptr;
  float *p_s2_  = nullptr;
  float *p_s3_  = nullptr;
  float *p_s4_  = nullptr;
  float *p_hjv_ = nullptr;
  int   *p_hji_ = nullptr;
};

#endif  // EKF_ACCEL_HPP_
