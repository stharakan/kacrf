
#ifndef MIXER_HPP
#define MIXER_HPP


#include <gofmm_interface.hpp>

namespace kacrf
{
	// Message mixing class
	class MessageMixer 
	{
		public:
		// Constructor
		MessageMixer(float _self, float _other)
		{
			this->self_weight = _self;
			this->oth_weight = _other;
		} // end constructor

		// mix appearance and spatial messages
		hData BinaryMix( hData mspa, hData mapp)
		{
			hData probs = mspa;

			// handle binary case
			int nn = (int) mspa.row();
			int cc = 2;

			// loop over pixels
			#pragma omp parallel for
			for (int ni = 0; ni < nn; ni++)
			{
				// get each class value
				float spa0 = mspa(ni,0);
				float spa1 = mspa(ni,1);
				float app0 = mapp(ni,0);
				float app1 = mapp(ni,1);

				// calculate passed values
				float t0 = this->self_weight * app0 + this->oth_weight * app1 + spa1; 
				float t1 = this->self_weight * app1 + this->oth_weight * app0 + spa0; 
				
				// set probs appropriately
				probs.setvalue(ni,0,t1); 
				probs.setvalue(ni,1,t0); 
			}

			return probs;

		}; // end binary mixing function

		hData MultiMix(hData mspa, hData mapp)
		{
			// get details
			hData probs = mspa;
			int nn = (int) mspa.row();
			int cc = (int) mspa.col();

			// loop over pixels
			#pragma omp parallel for
			for (int ni = 0; ni < nn; ni++)
			{
				float spasum = 0.0;
				float appsum = 0.0;
				// loop over classes and sum first
				for (int ci = 0; ci < cc; ci++)
				{
					spasum += mspa[ni+(nn*ci)];
					appsum += mapp[ni+(nn*ci)];
				}
				
				// loop again and set each to the difference
				for (int ci = 0; ci < cc; ci++)
				{
					float spaself = mspa[ni+(nn*ci)];
					float appself = mapp[ni+(nn*ci)];
					float mc = (this->self_weight * appself ) // appearance self penalty
						+ (this->oth_weight * (appsum-appself)) // appearance neighbor penalty
						+ (spasum-spaself) ; // spatial penalty
					probs.setvalue(ni,ci,mc);
				}

			}
			
			return probs;
		}; // end multi-class mixing function

		hData MixMessages(hData mspa, hData mapp)
		{

			int cc = (int) mspa.col();
			hData probs;

			// binary case is handled separately (avoids inner loop)
			if (cc == 2)
			{
				probs = this->BinaryMix(mspa,mapp);
			}
			else
			{
				probs = this->MultiMix(mspa,mapp);
			}

			return probs;

		};

		private:
			float self_weight;
			float oth_weight;

	}; // end class

}; //end namespace kacrf

#endif

