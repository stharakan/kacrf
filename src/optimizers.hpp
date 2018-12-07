
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <gofmm_interface.hpp>
#include <utilities.hpp>
#include <math.h> //exp, log
#include <Brain.hpp>
#include <PairwiseMessage.hpp>


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
				float t0 = probs(ni,0);
				float t1 = probs(ni,1);

				probs.setvalue(ni,0,t1); 
				probs.setvalue(ni,1,t0); 
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
	
	/** Update from dense crf without no unary */
	void NoUnaryCRFUpdate(hData& Qmat, hData& pm, hData unary)
	{
		int nc = hDataSize(Qmat);

		PottsMixture(pm);

		#pragma omp parallel for
		for (int i=0; i<nc; i++)
		{
			// Compute actual update
			//float temp = log(unary[i]) - pm[i];

			// Load into Qmat
			Qmat[i] = exp(- pm[i]);
		}

	}; // end dense crf update

	/** Update from dense crf */
	void DenseCRFUpdate(hData& Qmat, hData& pm, hData unary)
	{
		int nc = hDataSize(Qmat);

		PottsMixture(pm);

		#pragma omp parallel for
		for (int i=0; i<nc; i++)
		{
			// Compute actual update
			float temp = log(unary[i]) - pm[i];
			//Qmat[i] = temp;

			// Load into Qmat
			Qmat[i] = exp(temp);
		}
		//std::cout << "update pre exp" << std::endl;
		//preexp.Print();
		//
		//std::cout << "update post exp" << std::endl;
		//Qmat.Print();


	}; // end dense crf update

	// Run crf for a given set of kernels, length of dices determines iteration count
	hData RunCRF( hData & dice_scores, Brain2D im, PairwiseMessenger pm,float targ=0.0)
	{
		int crf_iters = (int) dice_scores.size() - 1;
		hData Q = im.Unary();
		im.PrintDiceScore(dice_scores,targ); 
	
		for(int iter = 0; iter< crf_iters; iter++)
		{
			// Get pairwise message
			hData m = pm.ComputeMessage(Q); 
			
			// Combine with unary for update
			DenseCRFUpdate(Q, m,im.Unary()); 
			
			// Normalize update
			NormalizeProbabilities(Q);
			
			// Evaluate accuracy and print 
			im.PrintDiceScore(dice_scores,iter+1,Q,targ); 
		}
	
		return Q;
	
	};

}; // end namespace kacrf

#endif
