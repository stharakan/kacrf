
#ifndef GOFMM_INTERFACE
#define GOFMM_INTERFACE


#include <gofmm.hpp>
#include <containers/SPDMatrix.hpp>
#include <containers/KernelMatrix.hpp>




namespace kacrf
{
	typedef hmlp::Data<float> hData;
	typedef hmlp::gofmm::Configuration<float> GoFMM_config;
	typedef hmlp::gofmm::randomsplit<hmlp::KernelMatrix<float>,2,float> GoFMM_RSPLIT;
	typedef hmlp::gofmm::centersplit<hmlp::KernelMatrix<float>,2,float> GoFMM_CSPLIT;
	typedef hmlp::Data< std::pair<float,size_t> > GoFMM_neighbors;
	typedef hmlp::KernelMatrix<float> GoFMM_Kernel;
	typedef hmlp::tree::Tree<hmlp::gofmm::Setup<GoFMM_Kernel, GoFMM_CSPLIT, float>, hmlp::gofmm::NodeData<float> > GoFMM_tree;

	/* Configure function -- default */
	GoFMM_config ConfigureGoFMM()
	{
 		/** [Required] Problem size. */
  	size_t n = 2500;
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
		/** Distance metric default is ANGLE_DISTANCE */
		//DistanceMetric metric = ANGLE_DISTANCE;
		DistanceMetric metric = GEOMETRY_DISTANCE;


		// Get config and return
		GoFMM_config config( metric, n, m, k, s, stol, budget );
		return config;


	};// end configure default
	
	
	
	/* Configure function -- takes params */
	GoFMM_config ConfigureGoFMM(DistanceMetric metric, size_t n, size_t m, size_t k, size_t s, float stol, float budget)
	{
		// Get config and return
		GoFMM_config config( metric, n, m, k, s, stol, budget );
		return config;
	};// end config w/ params

	
	/* Kernel class */
	class Kernel
	{
		public: 

		/* Default constructor */
		Kernel()
		{
			/* Do nothing */
		}
		
		/* Constructor -- sets up trees + finds neighbors*/
		Kernel( hData X, GoFMM_config conf)
		{
			// Set config
			this->config = conf;

			// Set up kernel
			hmlp::KernelMatrix<float> K( X );

			// Find tree splitters
			GoFMM_RSPLIT rsplitter( K );
			GoFMM_CSPLIT csplitter( K );

			// Find neighbors
  		auto neighbors = hmlp::gofmm::FindNeighbors( K, rsplitter, config); 
			
			// Compress
  		auto* tree_ptr = hmlp::gofmm::Compress( K, neighbors, csplitter, rsplitter, this->config );
			this->tree_ptr = tree_ptr;

			// Create ksum vector
			hData w(X.col(),1);
			std::fill(w.begin(), w.end(),1.0);
			this->ksum = this->Multiply(w);

		};//end kernel constructor


		/* Quick ksum print */
		void PrintKsum(){this->ksum.Print();};

		/* Normalized multiply */
		hData NormMultiply(hData w)
		{
			//TODO: add check for dimensionality
			
			// Dereference tree
			auto& tree = *(this->tree_ptr);

			// Compute multiply
  		auto u2 = hmlp::gofmm::Evaluate( tree, w );

			// loop over u2, 
			int ksize = this->ksum.size();
			#pragma omp parallel for
			for (int i =0; i< u2.size();i++)
			{
				// ASSUME GAUSSIAN KERNEL!!
				int ki = i % ksize;
				float denom = this->ksum[ki] - 1.0;
				// check for zero
				if (denom <= 1e-7 || denom >= -1e-7){denom=1.0;};
				
				float temp = (u2[i] - w[i])/(denom);
				u2[i] = temp;
			}
			return u2;
		};// End norm multiply function

		/* Multiply */
		hData Multiply(hData w)
		{
			//TODO: add check for dimensionality
			
			// Dereference tree
			auto& tree = *(this->tree_ptr);

			// Compute multiply
  		auto u2 = hmlp::gofmm::Evaluate( tree, w );
			return u2;


		};// End multiply function


		private:
			GoFMM_tree* tree_ptr;
			GoFMM_config config;
			hData ksum;


	};// end kernel class



};//end namespace kacrf

#endif /*end of gofmm_interface*/
