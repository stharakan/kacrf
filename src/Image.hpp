
#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <gofmm_interface.hpp>
#include <utilities.hpp>


namespace kacrf
{
	/* Image class, 2d for now */
	class Image
	{
		public:
			/** default constructor */
			Image()
			{
				/** Do nothing */
			};

			/** constructor from size */
			Image(size_t imsz)
			{
				// Initialize data to im size
				hData curim( imsz, imsz ); 
				curim.randn(1.0,1.0); // normally distributed image, mean 1.0, std 1.0

				// Add separator value to central square
				ModifyImageCenter(curim,3.0,1.0); 

				// set curim to image
				this->image = curim;
				this->imsz = (int) imsz;
			};

			/** TODO constructor from file (for brains) */

			/** TODO Spatial feature function */
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

				// for now nothing -- for compilation
				//hData temp(3,3);
				//return temp;
			};
			
			/** TODO Appearance feature function */
			hData ExtractAppearanceFeatures(float app_bw_spa, float app_bw_int)
			{
				int nn = this->pixels();
				int feats = 3;

				// initialize data
				hData fout(feats,nn); // positional features + appearance

				// Loop over data..
				#pragma omp parallel for
				for (int i =0; i < nn; i++)
				{
					// get indices
					int xidx = feats*i;
					int yidx = xidx + 1;
					int fidx = yidx + 1;

					// get vals
					float yval = i/this->imsz;
					float xval = i - (yval*this->imsz);

					fout[xidx] = xval / app_bw_spa;
					fout[yidx] = yval / app_bw_spa;
					fout[fidx] = this->image[i]/app_bw_int;
				}
				return fout;
				
				// for now nothing -- for compilation
				//hData temp(3,3);
				//return temp;
			};
			
			/** Easy func to print seg */
			void PrintImage(){ this->image.Print(); };
			
			/** Easy func to return imsize */
			int pixels(){ return (this->imsz*this->imsz); };
			//int pixels(){ return hDataSize(this->image); };
	

		private:
			hData image;
			int imsz;

	};// end class image


	/* Test image class -- includes probs and segmentation */
	class TestImage : public Image
	{
		public:
			/** default constructor */
			TestImage()
			{
				/** Do nothing */
			};

			/** constructor from size */
			TestImage(size_t imsz,float noise = 0.4) : Image(imsz)
			{
				// Initialize unary
				this->unary = DummyBinaryProbs(imsz,noise);
				
				// Initialize seg
				this->seg = DummyBinarySeg(imsz);
			};
			
			/** Easy func to print unary */
			void PrintUnary(){ this->unary.Print(); };

			/** Easy func to print seg */
			void PrintSeg(){ this->seg.Print(); };
			
			/** Easy func for accessing true seg */
			hData Unary(){ return this->unary;};

			/** Easy func for accessing true seg */
			hData Seg(){ return this->seg;};

			/** Easy func for unary seg */
			hData UnarySeg(){ return ProbabilityToSeg(this->unary); };
			
			/** Function to print dice score on first iter */
			void PrintDiceScore(std::vector<float>& dices,float target = 1.0)
			{
				hData yg = ProbabilityToSeg(this->unary); 
				float curdice = ComputeDice(this->seg,yg); //TODO, for now on floats --> soon template it

				// Print what we want -- 1. iter dice 2. 0 (init)
				std::cout << " It |   Dice   " << std::endl << "---------------" << std::endl;
				std::cout << 0 << " | " << curdice << std::endl;

				// Save to first element of dices, TODO -- check this with an assert?
				dices[0] = curdice;
			};

			void PrintDiceScore(std::vector<float>&dices, int iter, hData Qmat, float target = 1.0)
			{
				hData yg = ProbabilityToSeg(Qmat); 
				float curdice = ComputeDice(this->seg,yg); //TODO, for now on floats --> soon template it

				// Print what we want -- 1. iter dice 2. 0 (init)
				//std::cout << " It |   Dice   " << std::endl << "---------------" << std::endl;
				std::cout << iter << " | " << curdice << std::endl;

				// Save to first element of dices, TODO -- check this with an assert?
				dices[iter] = curdice;
			};


		private:
			hData unary;
			hData seg;
	};// End class TestImage






};//end namespace kacrf



#endif //end of IMAGE_HPP
