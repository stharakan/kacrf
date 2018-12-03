#include <iostream>
#include <fstream>
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
	size_t m = 128;
	sscanf( argv[ 2 ], "%lu", &m );
	/** [Required] Number of nearest neighbors. */
	size_t k = 64;
	sscanf( argv[ 3 ], "%lu", &k );
	/** Maximum off-diagonal rank (not used in neighbor search). */
	size_t s = 8;
	sscanf( argv[ 4 ], "%lu", &s );
	/** Approximation tolerance (not used in neighbor search). */
	float stol = 1E-6;
	sscanf( argv[ 5 ], "%f", &stol );
	/** The amount of direct evaluation (not used in neighbor search). */
	float budget = 0.9;
	sscanf( argv[ 6 ], "%f", &budget );
	/** App bw spa */
	float app_bw_spa = 20.0;
	//sscanf( argv[ 7 ], "%f", &app_bw_spa );

	/** app bw int */
	float app_bw_int = 1.0;
	//sscanf( argv[ 8 ], "%f", &app_bw_spa );

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
	hData Fapp = im.ExtractAppearanceFeatures(app_bw_spa,app_bw_int); 
	double feat_time = omp_get_wtime() - feat_time_beg;
	
	// Compute actual kernel
	double kern_time_beg = omp_get_wtime();
	Kernel kspa = Kernel(Fapp,config);
	double kern_time = omp_get_wtime() - kern_time_beg;

	
	size_t ngid = 1000;
	//auto amap = std::vector<size_t>(1, gid);
	//auto bmap = std::vector<size_t>(100);
	//for ( size_t j = 0; j< bmap.size(); j++) bmap[j] =j;
	//std::cout << " main call " <<std::endl;
	//kspa.PrintSources();
	//hData Kab = kspa.Ksub(amap,bmap);
	//Kab.Print();
	

	// multiply timings
	double mult_time = 0.0;
	int miters = 10;
	float err = 0.0;
	kacrf::Statistic mv_err;
	kacrf::Statistic mv_time;
	std::vector<float> errs(miters,0.0);

	// loop over miters
	for (int i = 0; i<miters; i++)
	{
		// initialize
		hData w(n,1);
		w.randn();
		//w[0] = 1.0;
		//for (int j = 1; j <w.size();j++){ w[j] = 0.0;}


		// multiply
		double i_mtime = omp_get_wtime();
		hData u = kspa.Multiply(w);
		mv_time.Update(omp_get_wtime() - i_mtime);

		// err compute
		float cerr = kspa.ComputeError(w,u,ngid);
		mv_err.Update(cerr);

	}

	int threads = omp_get_num_threads();
	std::cout << "-----------------------" << std::endl;
	std::cout << "    RESULT SUMMARY  " << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "    PARAMS  " << std::endl;
	std::cout << "N: "<< n << std::endl;
	std::cout << "imsz: "<< imsz << std::endl;
	std::cout << "m: "<< m << std::endl;
	std::cout << "k: "<< k << std::endl;
	std::cout << "s: "<< s << std::endl;
	std::cout << "stol: "<< stol << std::endl;
	std::cout << "budget: "<< budget << std::endl;
	std::cout << "threads: "<< threads << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "    TIMES  " << std::endl;
	std::cout << "Feat time: " << feat_time <<std::endl;
	std::cout << "Kern time: " << kern_time <<std::endl;
	std::cout << "Mult: " << mv_time._avg <<std::endl;
	mv_time.Print();
	std::cout << "-----------------------" << std::endl;
	std::cout << "   MATVEC ERR  " << std::endl;
	std::cout << "Err: " << mv_err._avg <<std::endl;
	mv_err.Print();
	std::cout << "-----------------------" << std::endl;

	/** file to append data to */ 
	std::string fname = "data_kapp.csv";
	std::ofstream outfile;
	outfile.open(fname, std::ios_base::app);
	outfile << mv_err._avg << "," << mv_err._max << "," << mv_err._min 
		<< "," << mv_time._avg << "," << mv_time._max << "," << mv_time._min
		<< "," << kern_time << "," 
		<< app_bw_spa << "," << app_bw_int << ","
		<< n << "," << m << "," << k << "," 
		<< s << "," << stol << "," << budget << std::endl;
	outfile.close();


	/** finalize hmlp */
	hmlp_finalize();

}
