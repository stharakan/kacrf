#include <iostream>
#include <hmlp.h>

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
	
	// Create image, probabilities, etc.
	std::cout << "Creating dummy problem .." << std::endl;

	// Initialize CRF 
	std::cout << "Initializing CRF .." << std::endl;

	// Run CRF
	std::cout << "Running CRF .. " << std::endl;

	// Save output
	std::cout << "Successful CRF run, saving output" << std::endl;
	
	// Initialize by calling hmlp api
	hmlp_finalize();
}


