
#ifndef EKF_HLS_H_
#define EKF_HLS_H_

// used to size the hardware
static constexpr int MAX_LANDMARKS = 100;
static constexpr int MAX_N_STATE = 2 * MAX_LANDMARKS + 3;  // 103
static constexpr int HJ_ACTIVE_COLS = 5;

// Buffer sizes 
static constexpr int MAX_SIGMA_SIZE = MAX_N_STATE * MAX_N_STATE;
static constexpr int HJ_VALS_SIZE = 2 * HJ_ACTIVE_COLS;
static constexpr int HJ_IDX_SIZE = HJ_ACTIVE_COLS;
static constexpr int MAX_PHT_SIZE = MAX_N_STATE * 2;

extern "C" {


void ekf_spmm(float *pht,
              const float *sigma0, const float *sigma1, const float *sigma2,
              const float *sigma3, const float *sigma4,
              const float *hj_vals,
              const int *hj_idx,
              const int n_state);
}

#endif  
