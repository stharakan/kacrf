#include <iostream>
//#include <hmlp.h>
//#include <gofmm/gofmm.hpp>
//#include <containers/KernelMatrix.hpp>
//#include <containers/VirtualMatrix.hpp>
#include <gofmm_interface.hpp>
#include <PairwiseMessage.hpp>

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
	DummyImage* im = DummyImage(imsz); //TODO --> in image.cpp
	

	// Create features
	hData* Fspa = im->ExtractSpatialFeatures(spa_bw); //TODO -->image.cpp
	hData* Fapp = im->ExtractAppearanceFeatures(app_bw_spa, app_bw_int); //TODO--> image.cpp
	
	Kernel kspa(Fspa, config);
	Kernel kapp = Kernel(Fapp,config); // appearance kernel
	
	// Initialize Pairwise messaging object
	PairwiseMessenger pm = PairwiseMessenger(kspa, kapp, app_weight,pair_weight); 
	
	/* CRF iterations */
	// Initial accuracy, print
	hData<float> dice_scores;
	hData<float> Q = im->unary;
	im->PrintDiceScore(dice_scores,0); //TODO also need to change this from im class

	for(int iter = 0; iter< 10; iter++)
	{
		// Get pairwise message
		hData m = pm.ComputeMessage(Q); //messenger class computes by calling Kernel multiplies
		
		// Combine with unary for update
		CRFUpdate(Q, m); // TODO - Q is updated within this func --> in optimizers.cpp for now?
		
		// Normalize update
		NormalizeProbabilities(Q); //TODO -- Q updated within Normalize Probabilities --> in optimizers.cpp for now?
		
		// Evaluate accuracy and print 
		im-> PrintDiceScore(dice_scores,iter+1,Q); //TODO - prints updated dice and loads into dice score --> in utilities.cpp
	}

	/** finalize hmlp */
	hmlp_finalize();

}
