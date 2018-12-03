
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
			this->X = X;

			// Set up kernel
			GoFMM_Kernel K( X );

			// Find tree splitters
			GoFMM_RSPLIT rsplitter( K );
			GoFMM_CSPLIT csplitter( K );

			// Find neighbors
  		auto neighbors = hmlp::gofmm::FindNeighbors( K, rsplitter, config); 
			
			// Compress
  		auto* tree_ptr = hmlp::gofmm::Compress( K, neighbors, csplitter, rsplitter, this->config );
			this->tree_ptr = tree_ptr;
			
			// Self testing --> just a check
			hmlp::gofmm::SelfTesting(* (this->tree_ptr) , 100, 10); 

			// Create ksum vector
			hData w(X.col(),1);
			std::fill(w.begin(), w.end(),1.0);
			this->ksum = this->Multiply(w);
		};//end kernel constructor

		/** pick out subset of index 1:ntot */
		static std::vector<size_t> RandomIndexSubset(size_t ntot, size_t nsel)
		{
			// Initialize
			std::vector<size_t> vtot(ntot);
	
			// set up big vec
			std::iota(vtot.begin(),vtot.end(),0);
	
			// shuffle
			std::random_shuffle(vtot.begin(),vtot.end());
	
			// set output
			std::vector<size_t> vout(vtot.begin(), vtot.begin() + nsel);
			return vout;
	
		};


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

		/* Self testing func from gofmm TODO -- does not work, seg faults.. something happens with tree ptr that is inaccurate.*/
		void SelfTest(size_t ntest = 100, size_t nhrs = 10)
		{
			auto & tree = *(this->tree_ptr);

			hmlp::gofmm::SelfTesting(tree, ntest, nhrs); 

		};

		/* Func to retrieve my kernel */
		GoFMM_Kernel GetMyKernel()
		{
			GoFMM_Kernel K(this->X);

			return K;
		};


		/* Compute Error */
		float ComputeError(hData w, hData pot, size_t ngid = 1)
		{
			// Retrieve kernel
			GoFMM_Kernel K = this->GetMyKernel();

			// make amap
			size_t totn = K.row();
			if (ngid > totn){ ngid = totn; }

			std::vector<size_t> amap = Kernel::RandomIndexSubset(totn, ngid);
			//auto amap = std::vector<size_t>(ngid,0);
			//if (ngid != 1)
			//{
			//	size_t stepsz = totn/ngid;
			//	for ( size_t j = 1; j< amap.size(); j++) amap[j] =j*stepsz;
			//}

			// make bmap
			auto bmap = std::vector<size_t>(K.col());
			for ( size_t j = 0; j< bmap.size(); j++) bmap[j] =j;

			// Extract relevant kernel bits 
			auto Kab = K(amap,bmap);

			// Extract relevant part of precomputed potentials
			auto all_pot = std::vector<size_t>(pot.col());
			for ( size_t j = 0; j< all_pot.size(); j++) all_pot[j] =j;
			auto potentials = pot(amap,all_pot);
			auto exact = potentials;
			
			// Exact multiply
			hmlp::xgemm
		  (
		    "N", "N",
		    Kab.row(), w.col(), w.row(),
		    1.0,   Kab.data(),   Kab.row(),
		             w.data(),     w.row(),
		    0.0, exact.data(), exact.row()
		  );     
			
			// Compute exact part norm
			auto nrm2 = hmlp_norm( exact.row(),  exact.col(), exact.data(), exact.row() );

			// Subtract and norm of diff
			for (size_t j = 0; j< exact.size(); j++)
			{
				potentials[j] -= exact[j];
			}
			auto err = hmlp::hmlp_norm( potentials.row(), potentials.col(), potentials.data(), potentials.row() ); 

			// return relative err
			return err / nrm2;	
		};

		/* kernel submatx */
		hData Ksub(std::vector<size_t>& amap, std::vector<size_t>& bmap)
		{
			// Retrieve kernel
			GoFMM_Kernel K = this->GetMyKernel();

			// Get submatrix and return
			auto Kab = K(amap,bmap);
			return Kab;
		};

		/* print sources, do we need this anymore?? TODO */
		void PrintSources()
		{
			// Retrieve kernel
			GoFMM_Kernel K = this->GetMyKernel();

			// Print sources
			std::cout << " sources " << std::endl;
			K.PrintSources();
		};


		private:
			hData X;
			GoFMM_tree* tree_ptr;
			GoFMM_config config;
			hData ksum;


	};// end kernel class



};//end namespace kacrf

#endif /*end of gofmm_interface*/
