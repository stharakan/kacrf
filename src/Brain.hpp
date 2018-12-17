
#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <gofmm_interface.hpp>
#include <utilities.hpp>

namespace kacrf
{
	/* Class for 2d brain problems */
	class Brain2D 
	{
		public:
			/** default constructor */
			Brain2D()
			{
				/** Do nothing */
			};

			/** normal constructor, reads from binary */
			Brain2D(string &_bdir, string &_bname, int _slice,
					size_t _cc = 2,size_t _imsz=240,size_t _mods = 4)
			{
				// Assign bdir, bname, slice to class attributes
				this->bdir = _bdir;
				this->bname = _bname;
				this->slice = _slice;
				this->imsz = _imsz; // assume d2 is same
				this->cc = _cc; 
				this->mods = _mods;

				// Construct filenames
				string base = this->BaseFileName();
				string imfile = base + "_allmods.bin";
				string segfile = base + "_seg.bin";
				string probfile = base + "_probs.bin";
			
				// For each file, check and load 
				int nn = this->pixels();
				this->allmods = BinaryToData(imfile,this->mods,nn);
				this->seg = BinaryToData(segfile,nn,1);
				this->unary= BinaryToData(probfile,nn,this->cc);
			};

			/** Extract spatial features -- need to format this for gofmm */
			hData ExtractSpatialFeatures(float spa_bw)
			{
				int nn = this->pixels();

				// initialize data
				hData fout(2,nn); // positional features

				// Loop over data..
				#pragma omp parallel for
				for (int i =0; i < nn; i++)
				{
					// get indices
					int xidx = 2*i;
					int yidx = 2*i + 1;

					// get vals
					float yval = i/this->imsz;
					float xval = i - (yval*this->imsz);

					fout[xidx] = xval / spa_bw;
					fout[yidx] = yval / spa_bw;
				}
				return fout;

			};


			/** Extract appearance features -- need to format this for gofmm */
			hData ExtractAppearanceFeatures(float app_bw_spa, float app_bw_int)
			{
				int nn = this->pixels();
				int feats = this->mods + 2;

				// initialize data
				hData fout(feats,nn); // positional features + appearance

				// Loop over data..
				#pragma omp parallel for
				for (int i =0; i < nn; i++)
				{
					// get indices
					int xidx = feats*i;
					int yidx = xidx + 1;
					int t1idx = yidx + 1;
					int t1cidx= t1idx + 1;
					int t2idx = t1cidx + 1;
					int flidx = t2idx + 1;

					// get vals
					float yval = i/this->imsz;
					float xval = i - (yval*this->imsz);

					fout[xidx] = xval / app_bw_spa;
					fout[yidx] = yval / app_bw_spa;
					fout[t1idx] = this->allmods(0,i)/app_bw_int;
					fout[t1cidx]= this->allmods(1,i)/app_bw_int;
					fout[t2idx] = this->allmods(2,i)/app_bw_int;
					fout[flidx] = this->allmods(3,i)/app_bw_int;
				}
				return fout;


			};

			/** Return base file name -- used to open allmods, seg, etc. */
			string BaseFileName()
			{
				return this->bdir + "/" + this->bname + "_s" + std::to_string(this->slice);
			};
			
			/** Easy func to return imsize */
			int pixels(){ return (this->imsz*this->imsz); };

			/** Easy func to print unary */
			void PrintUnary(){ this->unary.Print(); };

			/** Easy func to print seg */
			void PrintSeg(){ this->seg.Print(); };
			
			/** Easy func for accessing true seg */
			hData Unary(){ return this->unary;};

			/** Easy func for accessing true seg */
			hData Seg(){ return this->seg;};
			
			/** Easy func for accessing true seg */
			hData Image(){ return this->allmods;};

			/** Easy func for unary seg */
			hData UnarySeg(){ return ProbabilityToSeg(this->unary); };
			
			void PrintDiceScore(std::vector<float>& dices,float target = 1.0)
			{
				hData yg = ProbabilityToSeg(this->unary); 
				float curdice = ComputeDice(this->seg,yg,target); 

				// Print what we want -- 1. iter dice 2. 0 (init)
				std::cout << " It |   Dice   " << std::endl << "---------------" << std::endl;
				std::cout << 0 << " | " << curdice << std::endl;

				// Save to first element of dices, 
				dices[0] = curdice;
			};

			void PrintDiceScore(std::vector<float>&dices, int iter, hData Qmat, float target = 1.0)
			{
				hData yg = ProbabilityToSeg(Qmat); 
				float curdice = ComputeDice(this->seg,yg,target); 

				// Print what we want -- 1. iter dice 2. 0 (init)
				//std::cout << " It |   Dice   " << std::endl << "---------------" << std::endl;
				std::cout << iter << " | " << curdice << std::endl;

				// Save to nth element of dices, 
				dices[iter] = curdice;
			};

			void SaveQ(hData Q, int iter)
			{


			};

		private:
			string bdir;
			string bname;
			int slice;
			int imsz;
			int cc;
			int mods;
			hData allmods;
			hData seg;
			hData unary;
 
	}; // end class Brain2D

}; // end kacrf


#endif
