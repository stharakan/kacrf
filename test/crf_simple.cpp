#include <iostream>
//#include <hmlp.h>
//#include <gofmm/gofmm.hpp>
//#include <containers/KernelMatrix.hpp>
//#include <containers/VirtualMatrix.hpp>
#include <gofmm_interface.hpp>
#include <PairwiseMessage.hpp>
#include <utilities.hpp>
#include <Image.hpp>

using namespace std;
using namespace kacrf;

//Simple testing class
int main( )
{
	/** [Required] Problem size. */
	size_t n = 100;
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

	/* Set CRF parameters*/
	size_t imsz = n*n; // image size
	float spa_bw = 1.0; // spatial bandwidth in spa kernel
	float app_bw_spa = 20; // spatial bandwidth in app kernel
	float app_bw_int = 0.5; // intensity bandwidth in app kernel
	float app_weight = 1.0; // weighting of app kernel
	float pair_weight = 100.0; // weighting of pairwise message
	
	/* Initialize HMLP */
	hmlp_init( );

	// Initialize config
	GoFMM_config config = ConfigureGoFMM();

	/* CRF setup */
	// Make image
	TestImage im = TestImage(imsz); 
	

	// Create features
	hData Fspa = im->ExtractSpatialFeatures(spa_bw); 
	hData Fapp = im->ExtractAppearanceFeatures(app_bw_spa, app_bw_int); 
	
	Kernel kspa(Fspa, config);
	Kernel kapp = Kernel(Fapp,config); // appearance kernel
	
	// Initialize Pairwise messaging object
	PairwiseMessenger pm = PairwiseMessenger(kspa, kapp, app_weight,pair_weight); 
	
	/* CRF iterations */
	// Initial accuracy, print
	hData<float> dice_scores;
	hData<float> Q = im->unary;
	im.PrintDiceScore(dice_scores); 


	for(int iter = 0; iter< 10; iter++)
	{
		// Get pairwise message
		hData m = pm.ComputeMessage(Q); 
		
		// Combine with unary for update
		CRFUpdate(Q, m,im->unary); 
		
		// Normalize update
		NormalizeProbabilities(Q);
		
		// Evaluate accuracy and print 
		im->PrintDiceScore(dice_scores,iter+1,Q); 
	}

	/** finalize hmlp */
	hmlp_finalize();

}
