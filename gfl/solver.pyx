

cdef extern from "cpp/include/graph_fl.h":
    int graph_fused_lasso(int, double*,
                            int, int*, int*,
                            double, double, double,
                            int, double ,
                            double*)
    
    int graph_fused_lasso_weight(int n, double* y, double* w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)

    int graph_fused_lasso_augmented(int n, double* y, double* w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double lambda2, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

def pygfl(long long n,
          double[:] y,
            long long ntrails,
            int[:] trails,
            int[:] breakpoints,
            double lam,
            double alpha,
            double inflate,
            long long maxsteps,
            double converge,
            double[:] beta
            ):
                
    cdef double* y_ptr = &y[0]
    cdef int* trails_ptr = &trails[0]
    cdef int* breakpoints_ptr = &breakpoints[0]
    cdef double* beta_ptr = &beta[0]
    
    
    graph_fused_lasso(<int> n,
                      y_ptr,
                      <int> ntrails,
                      <int*> trails_ptr,
                      <int*> breakpoints_ptr,
                      lam,
                      alpha,
                      inflate,
                      <int> maxsteps,
                      converge,
                      beta_ptr)

def pygfl_weight(long long n,
                 double[:] y,
                 double[:] w,
	         long long ntrails,
	         int[:] trails,
	         int[:] breakpoints,
	         double lam,
	         double alpha,
	         double inflate,
	         long long maxsteps,
	         double converge,
	         double[:] beta):
                
    cdef double* y_ptr = &y[0]
    cdef double* w_ptr = &w[0]
    cdef int* trails_ptr = &trails[0]
    cdef int* breakpoints_ptr = &breakpoints[0]
    cdef double* beta_ptr = &beta[0]
    
    
    graph_fused_lasso_weight(<int> n,
                      y_ptr,
                      w_ptr,
                      <int> ntrails,
                      <int*> trails_ptr,
                      <int*> breakpoints_ptr,
                      lam,
                      alpha,
                      inflate,
                      <int> maxsteps,
                      converge,
                      beta_ptr)    

def pygfl_augmented(long long n,
                 double[:] y,
                 double[:] w,
	         long long ntrails,
	         int[:] trails,
	         int[:] breakpoints,
	         double lam,
                 double lambda2, 
	         double alpha,
	         double inflate,
	         long long maxsteps,
	         double converge,
	         double[:] beta):
                
    cdef double* y_ptr = &y[0]
    cdef double* w_ptr = &w[0]
    cdef int* trails_ptr = &trails[0]
    cdef int* breakpoints_ptr = &breakpoints[0]
    cdef double* beta_ptr = &beta[0]
    
    
    graph_fused_lasso_augmented(<int> n,
                      y_ptr,
                      w_ptr,
                      <int> ntrails,
                      <int*> trails_ptr,
                      <int*> breakpoints_ptr,
                      lam,
                      lambda2,
                      alpha,
                      inflate,
                      <int> maxsteps,
                      converge,
                      beta_ptr)          
