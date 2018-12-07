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
	/* GoFMM Parameters -- fixed for now */
	size_t n = 240*240; // problem size
	size_t m = 4096; // maximum leaf node size
	size_t k = 128; // neighbors
	size_t s = 4096; // max off diag rank
	float stol = 1E-5; // approximation tol
	float budget = 0.5; // amount of direct eval
	DistanceMetric metric = GEOMETRY_DISTANCE; // metric

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
	hData Qf = RunCRF(dice_scores,im,pm,targ);


	string str = im.BaseFileName() + "_i" + std::to_string(crf_iters) + ".bin";
	std::cout << str<< " "<< Qf.size() << " " << sizeof(float) << std::endl;
	//std::ofstream file(str,std::ofstream::binary);
	//file.write( reinterpret_cast<const char*>( Qf.data() ), Qf.size() * sizeof(float) );
	//file.close();
	DataToBinary(Qf,str);

	std::cout << "-----------------------" << std::endl;
	std::cout << "    RESULT SUMMARY  " << std::endl;
	std::cout << "-----------------------" << std::endl;
	PrintConfigInfo(config);
	std::cout << "-----------------------" << std::endl;
	std::cout << "    DICE  " << std::endl;
	dice_scores.Print();

	/** finalize hmlp */
	hmlp_finalize();


}
