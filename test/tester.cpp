#include <iostream>
//#include <hmlp.h>
#include <gofmm/gofmm.hpp>
#include <containers/KernelMatrix.hpp>
//#include <containers/VirtualMatrix.hpp>


using namespace std;

//Simple testing class
int main( )
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
	size_t nrhs = 10;
	/** Regularization for the system (K+lambda*I). */
	float lambda = 1.0;	

	// Initialize by calling hmlp api
	hmlp_init( );

	// other parameters needed
	float h = 1.0;
	size_t d = 6;
	const bool ADAPTIVE = true;
	const bool LEVELRESTRICTION = false;
	//DistanceMetric metric = ANGLE_DISTANCE;

	// Data
	hmlp::Data<float> X( d, n ); 
	X.randn();

	// Neighbor list
	hmlp::Data<std::pair<float, std::size_t>> NN;

	// Set up kernel
	kernel_s<float> kernel;
	kernel.type = KS_GAUSSIAN;
	kernel.scal = -0.5 / (h * h);

	// Virtually initialize kernel matrix
	hmlp::KernelMatrix<float> K1( n,n,d,kernel,X );


	// Configure gofmm
	hmlp::gofmm::Configuration<float> config( ANGLE_DISTANCE, n, m, k, s, stol, budget );

	// Rename tree splitting classes
	using RKDTSPLITTER = hmlp::gofmm::randomsplit<hmlp::KernelMatrix<float>,2,float>;
	using SPLITTER = hmlp::gofmm::centersplit<hmlp::KernelMatrix<float>,2,float>;

	// initialize random tree
	RKDTSPLITTER rkdtsplitter;
	rkdtsplitter.Kptr = &K1;
	rkdtsplitter.metric = ANGLE_DISTANCE;

	// initialize ball tree
	SPLITTER splitter;
	splitter.Kptr = &K1;
	splitter.metric = ANGLE_DISTANCE;
		  
	///** [Step#4] Perform the iterative neighbor search. */
	//auto neighbors1 = hmlp::gofmm::FindNeighbors( K1, rkdtsplitter, config1 );
	//
	///** [Step#5] Compress the matrix with an algebraic FMM. */
	//auto* tree_ptr1 = gofmm::Compress( K1, neighbors1, splitter, rkdtsplitter, config1 );

	auto* tree_ptr1 = hmlp::gofmm::Compress
		<ADAPTIVE,LEVELRESTRICTION,SPLITTER,RKDTSPLITTER,float>
		( &X,K1, NN, splitter, rkdtsplitter, config );
	auto& tree1 = *tree_ptr1;

	///** [Step#6] Compute an approximate MATVEC. */
	
	hmlp::Data<float> w1( nrhs, n ); w1.randn();
	
	auto u1 = hmlp::gofmm::Evaluate( tree1, w1 );
	//
	///** [Step#7] Factorization (HSS using ULV). */
	//gofmm::Factorize( tree1, lambda ); 

	//// Create image, probabilities, etc.
	//std::cout << "Creating dummy problem .." << std::endl;

	//// Initialize CRF 
	//std::cout << "Initializing CRF .." << std::endl;

	//// Run CRF
	//std::cout << "Running CRF .. " << std::endl;

	//// Save output
	cout << "Successful CRF run, saving output" << std::endl;
	
	// Finalize by calling hmlp api
	hmlp_finalize();
	
	cout << "Successful CRF run, saving output" << std::endl;

	return 0;
}


