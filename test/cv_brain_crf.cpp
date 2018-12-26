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
	float stol = 1E-7; // approximation tol
	float budget = 0.8; // amount of direct eval
	DistanceMetric metric = GEOMETRY_DISTANCE; // metric

	/* Set CRF parameters*/
	size_t imsz =(size_t) sqrt((float) n); // image size
	float spa_bw = 1.0; // spatial bandwidth in spa kernel
	sscanf( argv[ 1 ], "%f", &spa_bw );
	float app_bw_spa = 40; // spatial bandwidth in app kernel
	sscanf( argv[ 2 ], "%f", &app_bw_spa );
	float app_bw_int = 0.5; // intensity bandwidth in app kernel
	sscanf( argv[ 3 ], "%f", &app_bw_int );
	int crf_iters = 5; // crf iters to take
	float targ = 0.0; // target -- 0 means WT


	int num_weights = 5;
	std::vector<float> self_weights(num_weights);
	for (int wi = 0; wi < num_weights; wi++)
	{
		self_weights[wi] = pow(10.0,(float) (wi - 2) );
	}
	std::vector<float> other_weights = self_weights;

	//float self_weights[12] = {  1000.0,100.0,10.0,1.0,  1000.0,100.0,10.0,1.0, 1000.0,100.0,10.0,1.0 }; // weighting of app kernel
	//float other_weights[12]= {  .001,.01,0.1,1.0,       .01,0.1,1.0,10.0,      0.1,1.0,10.0,100.0 }; // weighting of pairwise message
	//float self_weights[5]= {100.0,10.0,1.0,0.1,0.01}; // weighting of app kernel
	//float other_weights[5] = {1.0,1.0,1.0,1.0,1.0 }; // weighting of pairwise message
	
	/* Set brain parameters */
	string bdir = "/home1/03158/tharakan/research/kacrf/data/";
	//string bname = "Brats17_TCIA_621_1";
	//sscanf( argv[ 4 ], "%s", &bname );
	string bname = argv[4];
	int slc = 53;
	sscanf( argv[ 5 ], "%d", &slc );
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
	double kern_time_beg = omp_get_wtime();
	hData Fspa = im.ExtractSpatialFeatures(spa_bw); 
	hData Fapp = im.ExtractAppearanceFeatures(app_bw_spa, app_bw_int);  
	
	Kernel kspa = Kernel(Fspa,config);
	Kernel kapp = Kernel(Fapp,config); // appearance kernel
	double kern_time = omp_get_wtime() - kern_time_beg;
		


	//hData w(n,1);
	//w.randn();
	//hData u = kspa.Multiply(w);
	//float spa_err = kspa.ComputeError(w,u,ngid);
	//u = kapp.Multiply(w);
	//float app_err = kapp.ComputeError(w,u,ngid);
	

	for (int c = 0; c < num_weights*num_weights; c++)
	{
		double i_mtime = omp_get_wtime();
		int si = c % num_weights;
		int oi = c /num_weights; 

	
		// weights
		float self_weight = self_weights[si];
		float other_weight = other_weights[oi];

		// Initialize Pairwise messaging object
		PairwiseMessenger pm = PairwiseMessenger(kspa, kapp, self_weight,other_weight); 
		
		/* CRF iterations */
		// Initial accuracy, print
		hData dice_scores(crf_iters+1,1);
		hData Qf = RunCRF(dice_scores,im,pm,targ);


		string str = im.BaseFileName() + "_i" + std::to_string(crf_iters) + "_a" + std::to_string( (int) log10(self_weight))
			+ "_p" + std::to_string((int) log10(other_weight)) + "_as" + std::to_string( (int) app_bw_spa ) + 
			+ "_ai" + std::to_string( (int) log2(app_bw_int) ) + ".bin";
		std::cout << str<< " "<< Qf.size() << " " << sizeof(float) << std::endl;
		//std::ofstream file(str,std::ofstream::binary);
		//file.write( reinterpret_cast<const char*>( Qf.data() ), Qf.size() * sizeof(float) );
		//file.close();
		DataToBinary(Qf,str);
		float crf_time = omp_get_wtime() - i_mtime;

		std::cout << "-----------------------" << std::endl;
		std::cout << "    RESULT SUMMARY  " << std::endl;
		std::cout << "-----------------------" << std::endl;
		PrintConfigInfo(config);
		std::cout << "-----------------------" << std::endl;
		std::cout << " DICE  self: " << self_weight << " other: " << other_weight << std::endl;
		dice_scores.Print();
		std::cout << "-----------------------" << std::endl;
		std::cout << " TIME: " << std::endl;
		std::cout << "Kern: "<< kern_time << std::endl;
		std::cout << "CRF : "<< crf_time << std::endl;
		std::cout << "-----------------------" << std::endl;
		
		/** file to append data to */ 
		std::string fname = bdir + "data_cvcrf.csv";
		std::cout << fname << std::endl;
		std::ofstream outfile;
		outfile.open(fname, std::ios_base::app);
		outfile << bname 
			<< "," << slc 
			<< "," << dice_scores[0] 
			<< "," << dice_scores[1]
			<< "," << dice_scores[crf_iters]
			<< "," << spa_bw
			<< "," << app_bw_spa
			<< "," << app_bw_int
			<< "," << self_weight
			<< "," << other_weight
			<< "," << kern_time 
			<< "," << crf_time 
			<< std::endl;

		outfile.close();
	}

	/** finalize hmlp */
	hmlp_finalize();


}
