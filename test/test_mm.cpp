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
#include <MessageMixer.hpp>
#include <optimizers.hpp>
#include <math.h> //sqrt

using namespace std;
using namespace kacrf;

//Simple testing class
int main( int argc, char *argv[] )
{
	/** [Required] Problem size. */
	size_t n = 10000;
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
	/** Metric style */
	DistanceMetric metric2 = GEOMETRY_DISTANCE;


	/* Initialize HMLP */
  hmlp_init( &argc, &argv );


	/* Binary testing computation */
	// Initialize matrices
	hData mspa(n,2);
	hData mapp(n,2);
	hData zmat(n,2);
	std::fill(zmat.begin(),zmat.end(),0.0 );

	// set up values
	float spa0 = 7.0;
	float app0 = 11.0;
	float spa1 = 5.0;
	float app1 = 13.0;
	
	// fill with values 
	for (int i = 0; i< n; i++)
	{
		// background class
		mspa.setvalue(i,0,spa0);
		mapp.setvalue(i,0,app0);

		// target class
		mspa.setvalue(i,1,spa1);
		mapp.setvalue(i,1,app1);
	}

	// Mix message -- test spa by setting
	// both weights to 0.0
	MessageMixer mm_testspa = MessageMixer(0.0,0.0);
	hData testspa = mm_testspa.MixMessages(mspa,mapp);
	
	// Mix message -- test app other weight by setting
	// mspa to zmat, self = 0.0, oth = 1.0 
	MessageMixer mm_appoth = MessageMixer(0.0,1.0);
	hData appoth = mm_appoth.MixMessages(zmat,mapp);

	// Mix message -- test app self weight by setting 
	// mspa to zmat, self = 1.0, oth = 0.0
	MessageMixer mm_appself = MessageMixer(1.0,0.0);
	hData appself = mm_appself.MixMessages(zmat,mapp);

	// Mix messages -- test app with multiple weights
	// self = 1.0, oth = 2.0
	MessageMixer mm_all = MessageMixer(1.0,2.0);
	hData all = mm_all.MixMessages(mspa, mapp);

	/* Compute errors  */
	float errspa = 0.0;
	float erroth = 0.0;
	float errself = 0.0;
	float errall = 0.0;

	for (int i = 0; i < n; i ++)
	{
		// 0 class error
		errspa += testspa(i,0) - spa1;
		erroth += appoth(i,0) - app1;
		errself+= appself(i,0) - app0;
		errall += all(i,0) - ( 1.0 * app0  + 2.0 * app1 + spa1  );
	
		// 1 class error
		errspa += testspa(i,1) - spa0;
		erroth += appoth(i,1) - app0;
		errself+= appself(i,1) - app1;
		errall += all(i,1) - ( 1.0 * app1  + 2.0 * app0 + spa0  );
	}

	errspa /= 2*n;
	erroth /= 2*n;
	errself /= 2*n;
	errall /= 2*n;

	// Print errors
	std::cout << "-----------------------" << std::endl;
	std::cout << "    RESULT SUMMARY  " << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "spa : "<<errspa << std::endl;
	std::cout << "oth : "<<erroth << std::endl;
	std::cout << "self: "<<errself<< std::endl;
	std::cout << "all : "<<errall << std::endl;
	std::cout << "-----------------------" << std::endl;
	

	/** finalize hmlp */
	hmlp_finalize();

}
