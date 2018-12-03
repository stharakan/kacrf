
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

			// TODO
			/** normal constructor, reads from binary */
			Brain2D(string &bdir, string &bname, int slice)
			{
				// Assign bdir, bname, slice to class attributes
				// Construct file 
				// Append allmods.bin, 
				// Append seg.bin, 
				// Append probs.bin, 
				// For each file, check and load
			};

			//TODO
			/** Extract spatial features -- need to format this for gofmm */
			hData ExtractSpatialFeatures(float spa_bw)
			{


			};


			//TODO
			/** Extract appearance features -- need to format this for gofmm */
			hData ExtractAppearanceFeatures(float app_bw_spa, float app_bw_int)
			{

			};

			//TODO
			/** Return base file name -- used to open allmods, seg, etc. */
			string BaseFileName()
			{

			};

		private:
			string bdir;
			string bname;
			int slice;
			hData allmods;
			hData seg;
			hData probs;
 
	}; // end class Brain2D

}; // end kacrf


#endif
