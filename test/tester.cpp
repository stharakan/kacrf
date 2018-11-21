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
	sscanf( argv[ 1 ], "%lu", &n );
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
	DistanceMetric metric = ANGLE_DISTANCE;
	DistanceMetric metric2 = GEOMETRY_DISTANCE;


	/* Set CRF parameters*/
	size_t imsz =(size_t) sqrt((float) n); // image size
	float spa_bw = 1.0; // spatial bandwidth in spa kernel
	float app_bw_spa = 20; // spatial bandwidth in app kernel
	float app_bw_int = 0.5; // intensity bandwidth in app kernel
	float app_weight = 100.0; // weighting of app kernel
	float pair_weight = 1.0; // weighting of pairwise message
	int crf_iters = 10;
	
	//gofmm::CommandLineHelper cmd( argc, argv );
	
	/* Initialize HMLP */
  hmlp_init( &argc, &argv );

	// Initialize config TODO -- param version is needed!!
	//GoFMM_config config = ConfigureGoFMM();
	GoFMM_config config = ConfigureGoFMM(metric, n, m, k, s, stol, budget);
	GoFMM_config config2 = ConfigureGoFMM(metric2, n, m, k, s, stol, budget);

	/* CRF setup */
	// Make image
	TestImage im = TestImage(imsz); 
	hData imun = im.Unary();
	

	// Create features
	double feat_time_beg = omp_get_wtime();
	hData Fspa = im.ExtractSpatialFeatures(spa_bw); 
	hData Fapp = im.ExtractAppearanceFeatures(app_bw_spa, app_bw_int); 
	double feat_time = omp_get_wtime() - feat_time_beg;
	
	double kern_time_beg = omp_get_wtime();
	Kernel kspa = Kernel(Fspa,config2);
	Kernel kapp = Kernel(Fapp,config2); // appearance kernel

	// Test NormMultiply;
	//hData w1(n, 2); 
	//std::fill(w1.begin(), w1.end(),1.0);
	//hData out1 = kspa.Multiply(imun);
	//hData out2 = kspa.NormMultiply(imun);

	//std::cout << "ksum" << std::endl;
	//kspa.PrintKsum();
	//std::cout << "imun" << std::endl;
	//imun.Print();
	//std::cout << "unnormalized" << std::endl;
	//out1.Print();
	//std::cout << "normalized" << std::endl;
	//out2.Print();
	
	// Initialize Pairwise messaging object
	PairwiseMessenger pm = PairwiseMessenger(kspa, kapp, app_weight,pair_weight); 
	double kern_time = omp_get_wtime() - kern_time_beg;
	
	/* CRF iterations */
	// Initial accuracy, print
	hData dice_scores(crf_iters+1,1);
	hData Q = imun;
	//imun.Print();
	im.PrintDiceScore(dice_scores); 

	double it_time_beg = omp_get_wtime();
	for(int iter = 0; iter< crf_iters; iter++)
	{
		std::cout << "Iteration begin .." <<std::endl;
		dice_scores.Print();

		// Get pairwise message
		hData m = pm.ComputeMessage(Q); 
		//std::cout << "message print .." <<std::endl;
		//m.Print();
		
		// Combine with unary for update
		DenseCRFUpdate(Q, m,imun); 
		
		// Normalize update
		NormalizeLogProbabilities(Q);
		//std::cout << "q distribution print .." <<std::endl;
		//Q.Print();
		
		// Evaluate accuracy and print 
		im.PrintDiceScore(dice_scores,iter+1,Q); 
	}
	double it_time = omp_get_wtime() - it_time_beg;


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
	std::cout << "crf iters: "<< crf_iters << std::endl;
	std::cout << "threads: " << omp_get_num_threads() << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "    DICE  " << std::endl;
	dice_scores.Print();
	std::cout << "-----------------------" << std::endl;
	std::cout << "    TIMES  " << std::endl;
	std::cout << "Feat time: " << feat_time <<std::endl;
	std::cout << "Kern time: " << kern_time <<std::endl;
	std::cout << "Iter time: " << it_time <<std::endl;
	std::cout << "Tot  time: " << feat_time + kern_time + it_time << std::endl;
	std::cout << "-----------------------" << std::endl;


	/** finalize hmlp */
	hmlp_finalize();

}
