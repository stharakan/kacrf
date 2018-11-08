
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <gofmm_interface.hpp>
#include <utilities.hpp>
#include <math.h> //exp, log


namespace kacrf
{
	// Mix potts model
	void PottsMixture(hData& probs)
	{
		// handle binary case
		int nn = (int) probs.row();
		int cc = (int) probs.col();

		// binary case is handled separately (avoids inner loop)
		if (cc == 2)
		{
			// loop over pixels
			#pragma omp parallel for
			for (int ni = 0; ni < nn; ni++)
			{
				// switch classes
				float temp = probs[ni];
				probs[ni] = probs[ni + nn]; 
				probs[ni + nn] = temp;
			}

		}
		else
		{
			// loop over pixels
			#pragma omp parallel for
			for (int ni = 0; ni < nn; ni++)
			{
				float csum = 0.0;
				// loop over classes and sum first
				for (int ci = 0; ci < cc; ci++)
				{
					csum += probs[ni+(nn*ci)];
				}
				
				// loop again and set each to the difference
				for (int ci = 0; ci < cc; ci++)
				{
					float temp = probs[ni+(nn*ci)];
					probs[ni + (nn*ci)] = csum - temp;
				}

			}
		}
	}; // end of potts model

	/** Update from dense crf */
	void DenseCRFUpdate(hData& Qmat, hData& pm, hData unary)
	{
		int nc = hDataSize(Qmat);
		hData preexp(Qmat.row(), Qmat.col());

		PottsMixture(pm);

		#pragma omp parallel for
		for (int i=0; i<nc; i++)
		{
			// Compute actual update
			float temp = log(unary[i]) - pm[i];
			preexp[i] = temp;	
			Qmat[i] = temp;

			// Load into Qmat
			//Qmat[i] = exp(temp);
		}
		//std::cout << "update pre exp" << std::endl;
		//preexp.Print();
		//
		//std::cout << "update post exp" << std::endl;
		//Qmat.Print();


	}; // end dense crf update

}; // end namespace kacrf

#endif
