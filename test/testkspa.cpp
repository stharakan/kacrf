#include <iostream>
#include <stdio.h>
#include <omp.h>
//#include <hmlp.h>
//#include <gofmm/gofmm.hpp>
//#include <containers/KernelMatrix.hpp>
//#include <containers/VirtualMatrix.hpp>
#include <gofmm_interface.hpp>
#include <PairwiseMessage.hpp>
#include <utilities.hpp>
#include <Image.hpp>
#include <optimizers.hpp>
#include <math.h> //sqrt

using namespace std;
using namespace kacrf;

//Simple testing class
int main( int argc, char *argv[] )
{
	/** [Required] Problem size. */
	size_t n = 10000;
	//sscanf( argv[ 1 ], "%lu", &n );
	/** Maximum leaf node size (not used in neighbor search). */
	size_t m = 64;
	/** [Required] Number of nearest neighbors. */
	size_t k = 4;
	/** Maximum off-diagonal rank (not used in neighbor search). */
	size_t s = 2;
	/** Approximation tolerance (not used in neighbor search). */
	float stol = 1E-5;
	/** The amount of direct evaluation (not used in neighbor search). */
	float budget = 0.9;
	/** Number of right-hand sides. */
	size_t nrhs = 10;
	/** Regularization for the system (K+lambda*I). */
	float lambda = 1.0;	
	/** Metric style */
	DistanceMetric metric = GEOMETRY_DISTANCE;


	/* Set CRF parameters*/
	size_t imsz =(size_t) sqrt((float) n); // image size
	float spa_bw = 1.0; // spatial bandwidth in spa kernel
	
	//gofmm::CommandLineHelper cmd( argc, argv );
	
	/* Initialize HMLP */
  hmlp_init( &argc, &argv );

	// Initialize config 
	GoFMM_config config = ConfigureGoFMM(metric, n, m, k, s, stol, budget);

	// Make image
	TestImage im = TestImage(imsz); 
	hData imun = im.Unary();

	// Create features
	double feat_time_beg = omp_get_wtime();
	hData Fspa = im.ExtractSpatialFeatures(spa_bw); 
	double feat_time = omp_get_wtime() - feat_time_beg;
	
	double kern_time_beg = omp_get_wtime();
	Kernel kspa = Kernel(Fspa,config);
	double kern_time = omp_get_wtime() - kern_time_beg;

	// multiply timings
	double mult_time = 0.0;
	int miters = 5;
	for (int i = 0; i<miters; i++)
	{
		// initialize
		hData w(n,2);

		// multiply
		double i_mtime = omp_get_wtime();
		hData u = kspa.Multiply(w);
		mult_time += omp_get_wtime() - i_mtime;
	}
	
	mult_time = mult_time/(float) miters;

	// Gofmm self testing
	kspa.SelfTest();

	std::cout << "-----------------------" << std::endl;
	std::cout << "    RESULT SUMMARY  " << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "    PARAMS  " << std::endl;
	std::cout << "N: "<< n << std::endl;
	std::cout << "imsz: "<< imsz << std::endl;
	std::cout << "m: "<< m << std::endl;
	std::cout << "k: "<< k << std::endl;
	std::cout << "stol: "<< stol << std::endl;
	std::cout << "budget: "<< budget << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "    TIMES  " << std::endl;
	std::cout << "Feat time: " << feat_time <<std::endl;
	std::cout << "Kern time: " << kern_time <<std::endl;
	std::cout << "Mult time: " << mult_time <<std::endl;
	std::cout << "-----------------------" << std::endl;


	/** finalize hmlp */
	hmlp_finalize();

}
