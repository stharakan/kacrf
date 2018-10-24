
#ifndef PAIRWISE_HPP
#define PAIRWISE_HPP

#include <gofmm_interface.hpp>

namespace kacrf
{
	// Pairwise messaging class
	class PairwiseMessenger
	{
		public:
		// Constructor
		PairwiseMessenger( Kernel kspa, Kernel kapp, float aweight, float mweight)
		{
			// Save kernel pointers to current obj
			this-> kspa = kspa;
			this-> kapp = kapp;

			// Save weights
			this-> aweight = aweight;
			this-> mweight= mweight;
		};// end constructor
		
		/* Compute spatial message and normalize */
		hData ComputeSpatial(hData probs)
		{
			// Compute multiply
			hData m;
			m = this->kspa.Multiply(probs);

			// Normalize
			//kspa.DiagNormalize(m);

			return m;
		};

		/* Compute appearance message and normalize */
		hData ComputeAppearance(hData probs)
		{
			// Compute multiply
			hData m;
			m = this->kapp.Multiply(probs);

			// Normalize and subtract diagonal?
			//kapp.DiagNormalize(m);
			
			return m;
		}; 

		/* Combine spatial and appearance messages */
		hData CombineMessages(hData ma, hData ms)
		{
			// Find size of messages, initialize TODO: check that these are the same?
			int cc = (int) ma.row(); // number of classes
			int nn = (int) ma.col(); // number of rows
			hData m(cc,nn);

			// loop over nn, cc
			#pragma omp parallel for
			for  (int j = 0; j < nn; j++)
			{
				for (int i = 0; i < cc; i++)
				{
					int idx = j*nn + i;
					m[idx] = this->mweight * ( ms[idx] + this->aweight *  ma[idx] ); 
				}

			} 
			return m;

		}; //message combiner

		/* Function to compute multiplies and get final message */
		hData ComputeMessage(hData probs)
		{
			// Compute spatial message
			hData mspa = this->ComputeSpatial(probs);
			
			// Compute appearance message
			hData mapp = this->ComputeAppearance(probs);
			
			// Weight and add appropriately
			hData m = this->CombineMessages(mspa, mapp);
			return m;
		}; // message computation



		private:
		Kernel kspa, kapp;
		float aweight, mweight;
	};// end class PairwiseMessenger



};// end namespace kacrf

#endif // end pairwise_hpp
