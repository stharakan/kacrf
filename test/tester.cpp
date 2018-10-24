//#include <gofmm.hpp>
//#include <containers/SPDMatrix.hpp>
//#include <containers/KernelMatrix.hpp>
//#include <containers/VirtualMatrix.hpp>
#include <gofmm_interface.hpp>
#include <PairwiseMessage.hpp>

using namespace std;
using namespace kacrf;

//Simple testing class
int main( int argc, char *argv[] )
{
	/** [Required] Problem size. */
	size_t n = 5000;
	/** Maximum leaf node size (not used in neighbor search). */
	size_t m = 128;
	/** [Required] Number of nearest neighbors. */
	size_t k = 64;
	/** Maximum off-diagonal rank (not used in neighbor search). */
	size_t s = 128;
	/** Approximation tolerance (not used in neighbor search). */
	float stol = 1E-5;
	/** The amount of direct evaluation (not used in neighbor search). */
	float budget = 0.01;
	/** Number of right-hand sides. */
	size_t nrhs = 5;
	/** Regularization for the system (K+lambda*I). */
	float lambda = 1.0;	

	// Initialize by calling hmlp api
  hmlp_init( &argc, &argv );



	// other parameters needed
	float h = 1.0;
	size_t d = 4;
	size_t d2 = 10;

	// Feature sets, rand for now
	hmlp::Data<float> X1( d, n ); 
	X1.randn();
	hmlp::Data<float> X2( d2, n ); 
	X2.randn();
	//hmlp::hmlp_printmatrix( (int) n, (int) n, X.data(), sizeof(float) );

	// Initialize config
	GoFMM_config config = ConfigureGoFMM();

	// Initialize kernels
	Kernel Kspa(X1, config);
	Kernel Kapp(X2, config);
	
	// Compute message 
	hmlp::Data<float> w1(n,nrhs ); w1.randn();
	//auto u1 = K.Multiply(w1);
	PairwiseMessenger pm(Kspa,Kapp,1.5, 10.0);
	hData mm = pm.ComputeMessage(w1);


	// Neighbor list
	//hmlp::Data<std::pair<float, std::size_t>> NN;

	//// Virtually initialize kernel matrix
	//hmlp::KernelMatrix<float> K2( X );


	//// Configure gofmm
	//hmlp::gofmm::Configuration<float> config2( ANGLE_DISTANCE, n, m, k, s, stol, budget );

	//hmlp::gofmm::randomsplit<KernelMatrix<float>, 2, float> rkdtsplitter2( K2 );
  //hmlp::gofmm::centersplit<KernelMatrix<float>, 2, float> splitter2( K2 );
	//auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
	//
	//
	//auto* tree_ptr2 = gofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2 );
  //auto& tree2 = *tree_ptr2;

	//hmlp::Data<float> w1(n,nrhs ); w1.randn();
	//
	
	// Finalize by calling hmlp api
	hmlp_finalize();
	
	cout << "Successful CRF run, saving output" << std::endl;

	return 0;
}


