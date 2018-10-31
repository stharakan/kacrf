#ifndef KACRF_UTILITIES_HPP
#define KACRF_UTILITIES_HPP


#include <gofmm_interface.hpp>

#include <math.h> // sqrt
#include <random>
#include <iostream>

namespace kacrf
{
	const float EPS = 0.0000001; //1e-7

	/** Simple function to return hData size (m*n) as int */
	int hDataSize(hData in)
	{
		return (int) (in.row() * in.col());
	};


	/** Probability class list to segmentation, TODO templating? seg out as int? */
	hData ProbabilityToSeg(hData probs)
	{
		// Params, we assume row --> nn and col --> cc
		int nn = (int) probs.row();
		int cc = (int) probs.col();
		int imsz = sqrt(nn); 

		// initialize data
		hData seg(imsz,imsz);

		// loop over nn
		for (int j = 0; j < nn; j++)
		{
			float pmax = 0.0;
			int maxi = 0;
			// Sum current row, reset probability zeros
			for (int i = 0; i < cc; i++)
			{
				float curp = probs[j + i*nn];

				if (curp > pmax)
				{
					pmax = curp;
					maxi =  i;
				}
			}

			// load appropriate val into seg
			seg[j] = (float) maxi;
		}

		return seg;

	};

	/** Dice score computation */
	float ComputeDice(hData truth, hData guess, float target = 1.0)
	{
		// initialize
		int nn = hDataSize(truth);
		float tcount = 0.0; // count truth target hits
		float gcount = 0.0; // count guess target hits
		float icount = 0.0; // count intersection

		// loop
		for (int j = 0; j < nn; j++)
		{
			bool tbool = truth[j] == target;
			bool gbool = guess[j] == target;

			if (tbool){ tcount++;};
			
			if (gbool)
			{
				gcount++;
				if (tbool){ icount ++;};

			};
		}

		float dice = (2.0 * icount)/(tcount + gcount);
		return dice;
	};



	/** Simple loop function to modify middle of image to new distribution */
	void ModifyImageCenter(hData& im, float mu, float sig)
	{
		// Random number generator
		std::default_random_engine gen;
		std::normal_distribution<float> dist(mu,sig);
	
		// Parameter init
		int imsz = (int) im.col(); // number of rows, assume square
		int center_beg = imsz/5;
		int center_end = 4*imsz/5;
	
		#pragma omp parallel for
		for  (int j = center_beg; j < center_end ; j++) //col idx
		{
			for (int i = center_beg; i < center_end; i++) //row idx
			{
				im[j*imsz + i] = dist(gen);
			}
		} 
	};
	
	
	/** Simple loop function to modify middle of image to new value */
	void ModifyImageCenter(hData & im, float fillval)
	{
		// Parameter init
		int imsz = (int) im.col(); // number of rows, assume square
		int center_beg = imsz/5;
		int center_end = 4*imsz/5;
		int lda = sizeof(float);
	
		#pragma omp parallel for
		for  (int j = center_beg; j < center_end ; j++) //col idx
		{
			for (int i = center_beg; i < center_end; i++) //row idx
			{
				int idx = j*imsz + i;
				//std::cout << "Idx: " << idx << std::endl;
				im[idx] = fillval;
			}
		}
	}; // end fill func
	
	
	/** Function to Normalize probabilities -- modifies input */
	void NormalizeProbabilities(hData & probs)
	{
		// Params, we assume row --> nn and col --> cc
		int nn = (int) probs.row();
		int cc = (int) probs.col();
		
		// Loop over each pixel
		#pragma omp parallel for
		for (int j = 0; j < nn; j++)
		{
			float rowsum = 0.0;
			// Sum current row, reset probability zeros
			for (int i = 0; i < cc; i++)
			{
				float cval = probs[j + i*nn];
				if (cval < EPS)
				{
					probs[j + i*nn] = EPS;
					cval = EPS;
				}
				else if (cval > (1.0 - EPS))
				{
					probs[j + i*nn] = 1.0 - EPS;
					cval = 1.0 - EPS;
				}
	
	
	
				rowsum += cval;
			}
	
	
			// Divide current row, only if sum != 0
			float newsum = 0.0;
			if (rowsum == 0.0)
			{
				for (int i = 0; i < cc; i++)
				{
					probs[j+i*nn] = 1.0/( (float) cc);
					newsum += probs[j + i*nn];
				}
			}
			else
			{
				for (int i = 0; i < cc; i++)
				{
					probs[j+i*nn] = probs[j + i*nn] / rowsum;
					newsum += probs[j + i*nn];
				}
			}
		}
	
	
	
	
	}; // end probability normalizer
	
	
	/** Function to combine two probabilities */
	hData CombineBinaryProbabilities(hData un1, hData un2)
	{
		// initialize outprobs
		int pixels = hDataSize(un1);
		hData probs( (size_t) pixels, 2);
	
		// assert images are the same size TODO
	
		// loop over once
		for  (int j = 0; j < pixels ; j++) //col idx
		{
			probs[j] = un1[j];
			probs[j + pixels] = un2[j];
		}
	
		// return
		return probs;
	}; 

	/** Dummy binary probability creator */
	hData DummyBinaryProbs(size_t imsz,float noise = 0.4)
	{
		hData un1(imsz, imsz);
		un1.randn(0.25,noise);
		hData un0(imsz, imsz);
		un0.randn(0.75,noise);
		
		// Modify unary center
		ModifyImageCenter(un1,0.75,noise);
		ModifyImageCenter(un0,0.25,noise);
		
		// Resize and normalize probabilities
		hData unout = CombineBinaryProbabilities(un0,un1); 
		NormalizeProbabilities(unout);

		// return
		return unout;

	};


	/** Dummy binary seg creator */
	hData DummyBinarySeg(size_t imsz)
	{
		hData ss(imsz,imsz);
		ss.setvalue(0.0);

		// Modify seg and set
		ModifyImageCenter(ss,1.0);
		return ss;
	};
}; // namespace kacrf
#endif // end utilities

