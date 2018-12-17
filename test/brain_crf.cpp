#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <gofmm_interface.hpp>
#include <PairwiseMessage.hpp>
#include <utilities.hpp>
#include <optimizers.hpp>
#include <Image.hpp>
#include <Brain.hpp>

#include <math.h> //sqrt

using namespace std;
using namespace kacrf;

//Simple testing class
int main( int argc, char *argv[] )
{
	/** [Required] Problem size. */
	size_t n = 240*240;
	/** Maximum leaf node size (not used in neighbor search). */
	size_t m = 4096;
	/** [Required] Number of nearest neighbors. */
	size_t k = 128;
	/** Maximum off-diagonal rank (not used in neighbor search). */
	size_t s = 4096;
	/** Approximation tolerance (not used in neighbor search). */
	float stol = 1E-5;
	/** The amount of direct evaluation (not used in neighbor search). */
	float budget = 0.5;
	/** Metric style */
	DistanceMetric metric = GEOMETRY_DISTANCE;


	/* Set CRF parameters*/
	size_t imsz =(size_t) sqrt((float) n); // image size
	float spa_bw = 1.0; // spatial bandwidth in spa kernel
	float app_bw_spa = 40; // spatial bandwidth in app kernel
	float app_bw_int = 0.5; // intensity bandwidth in app kernel
	float app_weight = 100.0; // weighting of app kernel
	float pair_weight = .01; // weighting of pairwise message
	int crf_iters = 2; // crf iters to take
	float targ = 0.0; // target -- 0 means WT
	
	/* Set brain parameters */
	string bdir = "/home1/03158/tharakan/research/kacrf/data/";
	string bname = "Brats17_TCIA_621_1";
	int slc = 53;
	int cc = 2;
	int mods = 4;


	/* Initialize HMLP */
  hmlp_init( &argc, &argv );

	// Initialize config
	GoFMM_config config = ConfigureGoFMM(metric, n, m, k, s, stol, budget);

	/* CRF setup */
	// Make image
	Brain2D im = Brain2D(bdir,bname,slc); 
	
	//hData allmods = im.Image();
	//hData probs = im.Unary();
	//hData seg = im.Seg();
	//hData useg = im.UnarySeg();
	//std::cout << "Allmods size: " << allmods.row() << " by " << allmods.col() << std::endl;
	//std::cout << "T1  entry (136,110)/(110,136): " << allmods(0,136*240 + 110) << " / " << allmods(0,110*240 + 136) <<std::endl;
	//std::cout << "T1c entry (136,110)/(110,136): " << allmods(1,136*240 + 110) << " / " << allmods(1,110*240 + 136) <<std::endl;
	//std::cout << "T2  entry (136,110)/(110,136): " << allmods(2,136*240 + 110) << " / " << allmods(2,110*240 + 136) <<std::endl;
	//std::cout << "FL  entry (136,110)/(110,136): " << allmods(3,136*240 + 110) << " / " << allmods(3,110*240 + 136) <<std::endl;

	//std::cout << "Probs size: " << probs.row() << " by " << probs.col() << std::endl;
	//std::cout << "C0  entry (136,110)/(110,136): " << probs(136*240 + 110,0) << " / " << probs(110*240 + 136,0) <<std::endl;
	//std::cout << "C1  entry (136,110)/(110,136): " << probs(136*240 + 110,1) << " / " << probs(110*240 + 136,1) <<std::endl;


	//std::cout << "Seg size: " << seg.row() << " by " << seg.col() << std::endl;
	//std::cout << "Sg entry (136,110)/(110,136): " << seg[136*240 + 110] << " / " << seg[110*240 + 136] <<std::endl;
	//
	//std::cout << "USeg size: " << useg.row() << " by " << useg.col() << std::endl;
	//std::cout << "Usg entry(136,110)/(110,136): " << useg[136*240 + 110] << " / " << useg[110*240 + 136] <<std::endl;
	
	// Create features
	hData Fspa = im.ExtractSpatialFeatures(spa_bw); 
	hData Fapp = im.ExtractAppearanceFeatures(app_bw_spa, app_bw_int);  
	
	Kernel kspa = Kernel(Fspa,config);
	Kernel kapp = Kernel(Fapp,config); // appearance kernel
	
	// Initialize Pairwise messaging object
	PairwiseMessenger pm = PairwiseMessenger(kspa, kapp, app_weight,pair_weight); 
	
	/* CRF iterations */
	// Initial accuracy, print
	hData dice_scores(crf_iters+1,1);
	hData imu = im.Unary();
	hData initseg = ProbabilityToSeg(imu);
	hData Q = im.Unary();
	im.PrintDiceScore(dice_scores,targ); 

	for(int iter = 0; iter< crf_iters; iter++)
	{
		// Get pairwise message
		hData m = pm.ComputeMessage(Q); 
		
		std::cout << "m size: " << m.row() << " by " << m.col() << std::endl;
		std::cout << "m0 entry (136,110)/(110,136): " << m(136*240 + 110,0) << " / " << m(110*240 + 136,0) <<std::endl;
		std::cout << "m1 entry (136,110)/(110,136): " << m(136*240 + 110,1) << " / " << m(110*240 + 136,1) <<std::endl;
		
		// Combine with unary for update
		DenseCRFUpdate(Q, m,im.Unary()); 
		
		std::cout << "Q size: " << Q.row() << " by " << Q.col() << std::endl;
		std::cout << "Q0 entry (136,110)/(110,136): " << Q(136*240 + 110,0) << " / " << Q(110*240 + 136,0) <<std::endl;
		std::cout << "Q1 entry (136,110)/(110,136): " << Q(136*240 + 110,1) << " / " << Q(110*240 + 136,1) <<std::endl;

		// Normalize update
		NormalizeProbabilities(Q);
		
		std::cout << "Q size: " << Q.row() << " by " << Q.col() << std::endl;
		std::cout << "Q0 entry (136,110)/(110,136): " << Q(136*240 + 110,0) << " / " << Q(110*240 + 136,0) <<std::endl;
		std::cout << "Q1 entry (136,110)/(110,136): " << Q(136*240 + 110,1) << " / " << Q(110*240 + 136,1) <<std::endl;
		std::cout << "Un size: " << imu.row() << " by " << imu.col() << std::endl;
		std::cout << "Un0 entry (136,110)/(110,136): " << imu(136*240 + 110,0) << " / " << imu(110*240 + 136,0) <<std::endl;
		std::cout << "Un1 entry (136,110)/(110,136): " << imu(136*240 + 110,1) << " / " << imu(110*240 + 136,1) <<std::endl;
	
		// Evaluate accuracy and print 
		im.PrintDiceScore(dice_scores,iter+1,Q,targ); 
		//std::cout << "ma size: " << ma.row() << " by " << ma.col() << std::endl;
		//std::cout << "ma0entry (136,110)/(110,136): " << ma(136*240 + 110,0) << " / " << ma(110*240 + 136,0) <<std::endl;
		//std::cout << "ma1entry (136,110)/(110,136): " << ma(136*240 + 110,1) << " / " << ma(110*240 + 136,1) <<std::endl;
		//std::cout << "ms size: " << ms.row() << " by " << ms.col() << std::endl;
		//std::cout << "ms0entry (136,110)/(110,136): " << ms(136*240 + 110,0) << " / " << ms(110*240 + 136,0) <<std::endl;
		//std::cout << "ms1entry (136,110)/(110,136): " << ms(136*240 + 110,1) << " / " << ms(110*240 + 136,1) <<std::endl;
		//
		
	
	
		hData qseg = ProbabilityToSeg(Q);
		std::cout << "QSeg size: " << qseg.row() << " by " << qseg.col() << std::endl;
		std::cout << "Qsg entry(136,110)/(110,136): " << qseg[136*240 + 110] << " / " << qseg[110*240 + 136] <<std::endl;
		std::cout << "ISeg size: " << initseg.row() << " by " << initseg.col() << std::endl;
		std::cout << "Isg entry(136,110)/(110,136): " << initseg[136*240 + 110] << " / " << initseg[110*240 + 136] <<std::endl;
	}
	
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
	std::cout << "-----------------------" << std::endl;
	std::cout << "    DICE  " << std::endl;
	dice_scores.Print();

	/** finalize hmlp */
	hmlp_finalize();


}
