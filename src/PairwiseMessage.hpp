
#ifndef PAIRWISE_HPP
#define PAIRWISE_HPP

#include <gofmm_interface.hpp>
#include <MessageMixer.hpp>

namespace kacrf
{
	// Pairwise messaging class
	class PairwiseMessenger
	{
		public:
		// Constructor
		PairwiseMessenger( Kernel kspa, Kernel kapp, float sweight, float oweight)
		{
			// Save kernel pointers to current obj
			this-> kspa = kspa;
			this-> kapp = kapp;

			// Save weights
			this-> mm = MessageMixer(sweight,oweight);
		};// end constructor
		
		/* Compute spatial message and normalize */
		hData ComputeSpatial(hData probs)
		{
			// Compute multiply
			hData m;
			m = this->kspa.NormMultiply(probs);

			// Normalize
			//kspa.DiagNormalize(m);

			return m;
		};

		/* Compute appearance message and normalize */
		hData ComputeAppearance(hData probs)
		{
			// Compute multiply
			hData m;
			m = this->kapp.NormMultiply(probs);

			// Normalize and subtract diagonal?
			//kapp.DiagNormalize(m);
			
			return m;
		}; 

		/* Combine spatial and appearance messages */
		hData CombineMessages(hData ma, hData ms)
		{
			hData mout = this->mm.MixMessages(ma,ms);
			return mout;
		};//end combine messages

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
		MessageMixer mm;
	};// end class PairwiseMessenger



};// end namespace kacrf

#endif // end pairwise_hpp
