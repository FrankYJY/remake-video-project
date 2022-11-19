import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.lang.reflect.Array;
import java.util.Arrays;

import javax.swing.*;
// import java.awt.Color;
import javax.swing.text.StyledEditorKit.BoldAction;

// import java.lang.*;
// import org.opencv.imgproc.Imgproc;

public class VideoDisplay {

	JFrame frame;
	JLabel[] lbIm1List;
	BufferedImage[] processedImgBuffer;
	int width = 640; // default image width and height
	int height = 480;
	int layerLength = height*width;
	int layerLength2 = layerLength*2;
	int frameLength = layerLength*3;
	int frameNumber = 480;

	String foregroundFolderPath;
	String backgroundFolderPath;
	// mode 1: green screen for background on foreground
	// mode 0: static background on foreground
	int mode;

	public VideoDisplay(String[] args) {
		try{
			foregroundFolderPath = args[0];
			backgroundFolderPath = args[1];
			mode = Integer.valueOf(args[2]);
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	/** Read Image RGB
	 *  Reads the image of given width and height at the given imgPath into the provided BufferedImage.
	 */
	private byte[] readImageRGBOnceToExistArr(int width, int height, String imgPath)
	{
		try
		{
			File file = new File(imgPath);
			RandomAccessFile raf = new RandomAccessFile(file, "r");
			raf.seek(0);
			byte[] bytes = new byte[frameLength];

			raf.read(bytes);
			return bytes;
		}
		catch (FileNotFoundException e) 
		{
			e.printStackTrace();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		return null;
	}

	private void loadByteArrToBuffer(byte[] bytes, BufferedImage img){

		int ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				byte a = 0;
				byte r = bytes[ind];
				byte g = bytes[ind+layerLength];
				byte b = bytes[ind+layerLength2];

				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
//					int pix = ((a << 24) + (r << 16) + (g << 8) + b);
				img.setRGB(x,y,pix);
				ind++;
			}
		}

	}

	private float[] frameRGBToHSV(byte[] RGB){
		// H ∈ [0°, 360°], saturation S ∈ [0, 1], and value V ∈ [0, 1]
		// H ∈ [0, 1]
		float[] HSV = new float[frameLength];
		int ind = 0;
		float[] HSVpixel;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				int r = RGB[posInLayer1] & 0xff;
				int g = RGB[posInLayer2] & 0xff;
				int b = RGB[posInLayer3] & 0xff;
				HSVpixel = RGBtoHSBCalculation(r, g, b,null);
				HSV[posInLayer1] = HSVpixel[0];
				HSV[posInLayer2] = HSVpixel[1];
				HSV[posInLayer3] = HSVpixel[2];
				ind++;
			}
		}
		return HSV;
	}

	// HSB is HSV
	public static float[] RGBtoHSBCalculation(int r, int g, int b, float[] hsbvals) {
        float hue, saturation, brightness;
        if (hsbvals == null) {
            hsbvals = new float[3];
        }
        int cmax = (r > g) ? r : g;
        if (b > cmax) cmax = b;
        int cmin = (r < g) ? r : g;
        if (b < cmin) cmin = b;

        brightness = ((float) cmax) / 255.0f;
        if (cmax != 0)
            saturation = ((float) (cmax - cmin)) / ((float) cmax);
        else
            saturation = 0;
        if (saturation == 0)
            hue = 0;
        else {
            float redc = ((float) (cmax - r)) / ((float) (cmax - cmin));
            float greenc = ((float) (cmax - g)) / ((float) (cmax - cmin));
            float bluec = ((float) (cmax - b)) / ((float) (cmax - cmin));
            if (r == cmax)
                hue = bluec - greenc;
            else if (g == cmax)
                hue = 2.0f + redc - bluec;
            else
                hue = 4.0f + greenc - redc;
            hue = hue / 6.0f;
            if (hue < 0)
                hue = hue + 1.0f;
        }
        hsbvals[0] = hue;
        hsbvals[1] = saturation;
        hsbvals[2] = brightness;
        return hsbvals;
    }

	private int[] frameRGBByteToInt(byte[] RGB){
		// H ∈ [0°, 360°], saturation S ∈ [0, 1], and value V ∈ [0, 1]
		// H ∈ [0, 1]
		int[] res = new int[frameLength];
		int ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				res[posInLayer1] = RGB[posInLayer1] & 0xff;
				res[posInLayer2] = RGB[posInLayer2] & 0xff;
				res[posInLayer3] = RGB[posInLayer3] & 0xff;
				ind++;
			}
		}
		return res;
	}

	private byte[] frameRGBIntToByte(int[] RGB){
		// RGB must in int [0,255]
		byte[] res = new byte[frameLength];
		for (int i = 0; i < frameLength; i++) {
			res[i] = (byte)RGB[i];
		}
		return res;
	}

	// RGB        HSV
	// 25 125 38 [0.35500002, 0.8, 0.49019608]
	// 110 255 150 [0.37931037, 0.5686275, 1.0]
	// naive region
	// float[][] greenScreenRangeHSV = {
	// 	{(float)0.18, (float)0.20, (float)0.35}, //min
	// 	{(float)0.5, (float)1.0, (float)1.0} //max
	// };
	float[][][] greenScreenHSVRegions = {
		{
			{(float)0.16, (float)0.20, (float)0.35}, //min
			{(float)0.5, (float)1.0, (float)1.0} //max
		},
		{
			{(float)0.1, (float)0.1, (float)0.6}, //min
			{(float)0.44, (float)0.20, (float)1} //max
		},
		{
			{(float)0.32, (float)0.7, (float)0.27}, //min
			{(float)0.38, (float)1, (float)0.35} //max
		},
		{
			{(float)0.24, (float)0.3, (float)0.17}, //min
			{(float)0.52, (float)0.9, (float)0.35} //max
		},
		{
			{(float)0.15, (float)0, (float)0.45}, //min
			{(float)0.55, (float)0.2, (float)0.71} //max
		},
	};

	// just substitute
	private void extractAndReplaceGreenScreen(byte[] foreground, byte[] background, float[] foregroundHSV){
		// compare foregroundHSV with threshold, 
		//in green screen range - replace foreground with background
		//outside               - keep original
		int ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;

				// // naive region judge
				// if(greenScreenRangeHSV[0][0] <= foregroundHSV[posInLayer1] && foregroundHSV[posInLayer1] <= greenScreenRangeHSV[1][0] 
				// && greenScreenRangeHSV[0][1] <= foregroundHSV[posInLayer2] && foregroundHSV[posInLayer2] <= greenScreenRangeHSV[1][1]
				// && greenScreenRangeHSV[0][2] <= foregroundHSV[posInLayer3] && foregroundHSV[posInLayer3] <= greenScreenRangeHSV[1][2]
				// ){
				// 	foreground[posInLayer1] = background[posInLayer1];
				// 	foreground[posInLayer2] = background[posInLayer2];
				// 	foreground[posInLayer3] = background[posInLayer3];
				// 	// System.out.println("replacing");
				// }

				// iterate through regions
				for (int i = 0; i < greenScreenHSVRegions.length; i++) {
					if(greenScreenHSVRegions[i][0][0] <= foregroundHSV[posInLayer1] && foregroundHSV[posInLayer1] <= greenScreenHSVRegions[i][1][0] 
					&& greenScreenHSVRegions[i][0][1] <= foregroundHSV[posInLayer2] && foregroundHSV[posInLayer2] <= greenScreenHSVRegions[i][1][1]
					&& greenScreenHSVRegions[i][0][2] <= foregroundHSV[posInLayer3] && foregroundHSV[posInLayer3] <= greenScreenHSVRegions[i][1][2]
					){
						foreground[posInLayer1] = background[posInLayer1];
						foreground[posInLayer2] = background[posInLayer2];
						foreground[posInLayer3] = background[posInLayer3];
						// System.out.println("replacing");
						break;
					}
				}
				ind++;
			}
		}

	}

	// with linear edge
	private void extractAndReplaceGreenScreen(byte[] foreground, byte[] background, float[] foregroundHSV, int linearLength){
		// compare foregroundHSV with threshold, 
		//in green screen range - replace foreground with background
		//outside               - keep original

		// linearLength 3 0  0  0  1 1 1
        //               |--------| 
		//                0 1/3 2/3

		// also can do green subtraction on the edge by parameter if rgb

		int ind = 0;
		// 0 keep         1 substitute 
		float[] mask = new float[layerLength];
		Arrays.fill(mask, 0);
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				
				for (int i = 0; i < greenScreenHSVRegions.length; i++) {
					if(greenScreenHSVRegions[i][0][0] <= foregroundHSV[posInLayer1] && foregroundHSV[posInLayer1] <= greenScreenHSVRegions[i][1][0] 
					&& greenScreenHSVRegions[i][0][1] <= foregroundHSV[posInLayer2] && foregroundHSV[posInLayer2] <= greenScreenHSVRegions[i][1][1]
					&& greenScreenHSVRegions[i][0][2] <= foregroundHSV[posInLayer3] && foregroundHSV[posInLayer3] <= greenScreenHSVRegions[i][1][2]
					){
						mask[posInLayer1] = 1;
						break;
					}
				}
				
				ind++;
			}
		}

		ind = 0;
		for(int y = 0; y < height; y++)
		{
			ind = width * y;
			while(ind < width * y + width-1){
				if(mask[ind]==0 && mask[ind+1]==1){
					int ind2 = ind;
					while(ind2-1>=0 && ind - (ind2 - 1) < linearLength && mask[ind2-1]==0){
						ind2--;
					}
					int reallen = ind - ind2;
					if(reallen == linearLength-1){
						// skip hairs, not enough long, skip
						float interval = (float)1/linearLength;
						for (int i = 1; i < linearLength; i++) {
						   mask[ind2 + i] = interval * (float)i;						
						}
					}




				}else if(mask[ind+1]==0 && mask[ind]==1){


					int ind2 = ind + 1;
					while(ind2+1<width *(y+1) && ind2+1 - ind <= linearLength && mask[ind2+1]==0){
						ind2++;
					}
					int reallen = ind2 - ind;
					if(reallen == linearLength){
						float interval = (float)1/linearLength;
						for (int i = 1; i < linearLength; i++) {
						   mask[ind2 - i] = interval * (float)i;
						//    System.out.println(interval + " " + (float)i + " " + mask[ind + i]);
						}

					}

					ind = ind2;
				}
				ind++;
			}
			
		}

		ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;

				float bkgTense = mask[posInLayer1];
				float fgdTense = 1-bkgTense;
				// if(bkgTense !=0 && bkgTense!=1){
				// 	System.out.println("fraction" + bkgTense);
				// }
				// calcualte use int, back to byte
				foreground[posInLayer1] = (byte)( (background[posInLayer1] & 0xff)*bkgTense + (foreground[posInLayer1] & 0xff)*fgdTense );
				foreground[posInLayer2] = (byte)( (background[posInLayer2] & 0xff)*bkgTense + (foreground[posInLayer2] & 0xff)*fgdTense );
				foreground[posInLayer3] = (byte)( (background[posInLayer3] & 0xff)*bkgTense + (foreground[posInLayer3] & 0xff)*fgdTense );
				ind++;
			}
		}

	}

	int pastFrameCacheLength = 1;
	float[] HSVTolerateRange = {(float)0.005, (float)0.1, (float)0.1}; // tolerate a small swift on environment extraction, moving
	int[] RGBTolerateRange = {10, 10, 10};

	float[] HSVAvgTolerateRange = {(float)0.1, (float)0.7, (float)0.7}; // tolerate a small swift on environment extraction, moving
	int[] RGBAvgTolerateRange = {30, 30, 30};

	// HashMap<Integer, Integer>[] map = new HashMap<>()[frameLength];
	// count and update dict
	// dict ele count 0->1 ele add to set
	// dict ele count 1->0 ele delete from set
	// if we have integer val and do not use range

	float[][] pastForegroundCacheHSV;
	float[] currentOriginalHSVForeground;

	int[][] pastForegroundCacheRGB;
	int[] currentOriginalRGBForeground;

	boolean allowAvgExtract = true;
	boolean allowGaussianBlur = true;

	private void extractAndReplaceMovingHSV(byte[] foreground, byte[] background, float[] foregroundHSV, int outerI){
		if(outerI == 0){
			// first frame set 0
			Arrays.fill(foreground, (byte)0);
		}else{
			int validFrameCacheLength = Math.min(pastFrameCacheLength, outerI);
			int ind = 0;
			for(int y = 0; y < height; y++)
			{
				for(int x = 0; x < width; x++)
				{
					int posInLayer1 = ind;
					int posInLayer2 = ind+layerLength;
					int posInLayer3 = ind+layerLength2;

					for (int i = 0; i < validFrameCacheLength; i++) {
						float[] aPastFrame = pastForegroundCacheHSV[i];
						if(aPastFrame[posInLayer1] - HSVTolerateRange[0] < foregroundHSV[posInLayer1] && foregroundHSV[posInLayer1] < aPastFrame[posInLayer1] + HSVTolerateRange[0]
						&& aPastFrame[posInLayer2] - HSVTolerateRange[1] < foregroundHSV[posInLayer2] && foregroundHSV[posInLayer2] < aPastFrame[posInLayer2] + HSVTolerateRange[1]
						&& aPastFrame[posInLayer3] - HSVTolerateRange[2] < foregroundHSV[posInLayer3] && foregroundHSV[posInLayer3] < aPastFrame[posInLayer3] + HSVTolerateRange[2]
						){
							foreground[posInLayer1] = background[posInLayer1];
							foreground[posInLayer2] = background[posInLayer2];
							foreground[posInLayer3] = background[posInLayer3];
							// substitute once
							break;
						}
						
					}

					if(allowAvgExtract){
						// if in average range, substitute
						if(foregroundHSVAvg[posInLayer1] - HSVAvgTolerateRange[0] < foregroundHSV[posInLayer1] && foregroundHSV[posInLayer1] < foregroundHSVAvg[posInLayer1] + HSVAvgTolerateRange[0]
						&& foregroundHSVAvg[posInLayer2] - HSVAvgTolerateRange[1] < foregroundHSV[posInLayer2] && foregroundHSV[posInLayer2] < foregroundHSVAvg[posInLayer2] + HSVAvgTolerateRange[1]
						&& foregroundHSVAvg[posInLayer3] - HSVAvgTolerateRange[2] < foregroundHSV[posInLayer3] && foregroundHSV[posInLayer3] < foregroundHSVAvg[posInLayer3] + HSVAvgTolerateRange[2]
						){
							foreground[posInLayer1] = background[posInLayer1];
							foreground[posInLayer2] = background[posInLayer2];
							foreground[posInLayer3] = background[posInLayer3];
						}
					}


					ind++;
				}
			}
		}
	}

	int expandSize = 3;
	double sigma = 2;

	float[][] sigmaMatrix;
	private float[][] calculateGaussianMatrix(int expandSize, double sigma){
		// sigma: standard diviation
		int n = 2*expandSize-1;
		float[][] res = new float[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int u = Math.abs(expandSize-i-1);
				int v = Math.abs(expandSize-j-1);
				res[i][j] = (float)( 1.0/(2.0*Math.PI*Math.pow(sigma, 2)) * Math.exp(-(double)(Math.pow(u, 2)+Math.pow(v, 2))/(2.0 * Math.pow(sigma, 2))) );
			}
		}
		return res;
	}

	private int[] GaussianBlur(int[] original, float[][] inputSigmaMatrix){
		// original is RGB int 0-255
		int n = inputSigmaMatrix.length;
		int m = (n+1)/2-1;
		float[] blurred = new float[frameLength];
		Arrays.fill(blurred, 0);
		int ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				for(int j = 0; j < n; j++){
					for(int i = 0; i < n; i++){
						int curX = x - m + i;
						int curY = y - m + j;
						if(0<=curX && curX<width && 0<=curY && curY<height){
							int cur = curY*width+curX;
							blurred[cur] += original[posInLayer1]*inputSigmaMatrix[i][j];
							blurred[cur+layerLength] += original[posInLayer2]*inputSigmaMatrix[i][j];
							blurred[cur+layerLength2] += original[posInLayer3]*inputSigmaMatrix[i][j];
						}
					}
				}
				ind++;
			}
		}
		int[] blurredInt = new int[frameLength];
		// to int [0,255]
		ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				blurredInt[posInLayer1] = (int)Math.round(blurred[posInLayer1]);
				blurredInt[posInLayer2] = (int)Math.round(blurred[posInLayer2]);
				blurredInt[posInLayer3] = (int)Math.round(blurred[posInLayer3]);
				ind++;
			}
		}
		for (int i = 0; i < blurredInt.length; i++) {
			if(blurredInt[i] < 0){
				blurredInt[i] = 0;
			}else if(blurredInt[i] >255){
				blurredInt[i] = 255;
			}
		}
		return blurredInt;
	}

	private float[] GaussianBlur(float[] original, float[][] inputSigmaMatrix){
		// original is float
		int n = inputSigmaMatrix.length;
		int m = (n+1)/2-1;
		float[] blurred = new float[frameLength];
		Arrays.fill(blurred, 0);
		int ind = 0;
		for(int y = 0; y < height; y++)
		{
			for(int x = 0; x < width; x++)
			{
				int posInLayer1 = ind;
				int posInLayer2 = ind+layerLength;
				int posInLayer3 = ind+layerLength2;
				for(int j = 0; j < n; j++){
					for(int i = 0; i < n; i++){
						int curX = x - m + i;
						int curY = y - m + j;
						if(0<=curX && curX<width && 0<=curY && curY<height){
							int cur = curY*width+curX;
							blurred[cur] += original[posInLayer1]*inputSigmaMatrix[i][j];
							blurred[cur+layerLength] += original[posInLayer2]*inputSigmaMatrix[i][j];
							blurred[cur+layerLength2] += original[posInLayer3]*inputSigmaMatrix[i][j];
						}
					}
				}
				ind++;
			}
		}
		return blurred;
	}

	private void extractAndReplaceMovingRGB(byte[] foreground, byte[] background, int[] foregroundRGB, int outerI){
		if(outerI == 0){
			// first frame set 0
			Arrays.fill(foreground, (byte)0);
		}else{
			int validFrameCacheLength = Math.min(pastFrameCacheLength, outerI);
			int ind = 0;
			for(int y = 0; y < height; y++)
			{
				for(int x = 0; x < width; x++)
				{
					boolean skipflag = false;
					int posInLayer1 = ind;
					int posInLayer2 = ind+layerLength;
					int posInLayer3 = ind+layerLength2;

					if(allowAvgExtract){
						// if in average range, substitute
						if(foregroundRGBAvg[posInLayer1] - RGBAvgTolerateRange[0] < foregroundRGB[posInLayer1] && foregroundRGB[posInLayer1] < foregroundRGBAvg[posInLayer1] + RGBAvgTolerateRange[0]
						&& foregroundRGBAvg[posInLayer2] - RGBAvgTolerateRange[1] < foregroundRGB[posInLayer2] && foregroundRGB[posInLayer2] < foregroundRGBAvg[posInLayer2] + RGBAvgTolerateRange[1]
						&& foregroundRGBAvg[posInLayer3] - RGBAvgTolerateRange[2] < foregroundRGB[posInLayer3] && foregroundRGB[posInLayer3] < foregroundRGBAvg[posInLayer3] + RGBAvgTolerateRange[2]
						){
							foreground[posInLayer1] = background[posInLayer1];
							foreground[posInLayer2] = background[posInLayer2];
							foreground[posInLayer3] = background[posInLayer3];
							skipflag = true;
						}
					}

					if(!skipflag){
						for (int i = 0; i < validFrameCacheLength; i++) {
							int[] aPastFrame = pastForegroundCacheRGB[i];
							if(aPastFrame[posInLayer1] - RGBTolerateRange[0] < foregroundRGB[posInLayer1] && foregroundRGB[posInLayer1] < aPastFrame[posInLayer1] + RGBTolerateRange[0]
							&& aPastFrame[posInLayer2] - RGBTolerateRange[1] < foregroundRGB[posInLayer2] && foregroundRGB[posInLayer2] < aPastFrame[posInLayer2] + RGBTolerateRange[1]
							&& aPastFrame[posInLayer3] - RGBTolerateRange[2] < foregroundRGB[posInLayer3] && foregroundRGB[posInLayer3] < aPastFrame[posInLayer3] + RGBTolerateRange[2]
							){
								foreground[posInLayer1] = background[posInLayer1];
								foreground[posInLayer2] = background[posInLayer2];
								foreground[posInLayer3] = background[posInLayer3];
								// substitute once
								break;
							}
							
						}
					}




					ind++;
				}
			}
		}
	}

	void addToAvg(int[] foregroundRGB, int[] foregroundRGBAvg){
		for (int i = 0; i < frameLength; i++) {
			foregroundRGBAvg[i] += foregroundRGB[i];
		}
	}

	void divideAvg(int[] foregroundRGBAvg){
		for (int i = 0; i < frameLength; i++) {
			foregroundRGBAvg[i] /= frameNumber;
		}
	}

	private void updateCache(float[] currentOriginalForeground, int outerI){
		pastForegroundCacheHSV[outerI%pastFrameCacheLength] = currentOriginalForeground;
	}

	private void updateCache(int[] currentOriginalForeground, int outerI){
		pastForegroundCacheRGB[outerI%pastFrameCacheLength] = currentOriginalForeground;
	}

	int[] foregroundRGBAvg;
	byte[] foregroundRGBAvgInByte;
	float[] foregroundHSVAvg;

	int fps = 24;
	// int fps = 60;
	int linearLength = 2;

	boolean useHSV = false;
	boolean allowGrayScale = false;

	public synchronized void run(){
		// // Read a parameter from command line
		// String param1 = args[1];
		// System.out.println("The second parameter was: " + param1);

		// initialize buffers
		processedImgBuffer = new BufferedImage[frameNumber];

		int foregroundFolderPathSlashPos = Math.max(foregroundFolderPath.lastIndexOf('/'), foregroundFolderPath.lastIndexOf("\\"));
		int backgroundFolderPathSlashPos = Math.max(backgroundFolderPath.lastIndexOf('/'), backgroundFolderPath.lastIndexOf("\\"));
		String foregroundFolderPathLastPart = foregroundFolderPath.substring(foregroundFolderPathSlashPos + 1);
		String backgroundFolderPathLastPart = backgroundFolderPath.substring(backgroundFolderPathSlashPos + 1);
		String actualForegroundPath;
		String actualBackgroundPath;
		byte[] foreground = new byte[frameLength];
		byte[] background = new byte[frameLength];
		float[] foregroundHSV;
		float[] foregroundHSVBlurred;
		int[] foregroundRGB;
		int[] foregroundRGBBlurred = new int[frameLength];
		byte[] foregroundRGBBlurredByte;
		if(mode == 1){
			foregroundHSV = new float[frameLength];
			
		}else{
			// currentOriginalHSVForeground = new float[frameLength];
			pastForegroundCacheHSV = new float[pastFrameCacheLength][frameLength];
			pastForegroundCacheRGB = new int[pastFrameCacheLength][frameLength];
			currentOriginalRGBForeground = new int[frameLength];
			sigmaMatrix = calculateGaussianMatrix(expandSize, sigma);
			// for (int i = 0; i < sigmaMatrix.length; i++) {
			// 	System.out.println(Arrays.toString(sigmaMatrix[i]));
			// }
			// get average
			foregroundRGBAvg = new int[frameLength];
			for (int i = 0; i < frameNumber; i++) {
				actualForegroundPath = String.format("%s/%s.%04d.rgb", foregroundFolderPath, foregroundFolderPathLastPart, i);
				foreground = readImageRGBOnceToExistArr(width, height, actualForegroundPath);
				foregroundRGB = frameRGBByteToInt(foreground);
				addToAvg(foregroundRGB, foregroundRGBAvg);
			}
			divideAvg(foregroundRGBAvg);
			foregroundRGBAvgInByte = frameRGBIntToByte(foregroundRGBAvg);
			foregroundHSVAvg = frameRGBToHSV(foregroundRGBAvgInByte);
			
		}
		
		for (int i = 0; i < frameNumber; i++) {
			actualForegroundPath = String.format("%s/%s.%04d.rgb", foregroundFolderPath, foregroundFolderPathLastPart, i);
			actualBackgroundPath = String.format("%s/%s.%04d.rgb", backgroundFolderPath, backgroundFolderPathLastPart, i);
			// System.out.println(actualForegroundPath);
			// initialize buffer
			processedImgBuffer[i] = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			// all inplace
			foreground = readImageRGBOnceToExistArr(width, height, actualForegroundPath);
			background = readImageRGBOnceToExistArr(width, height, actualBackgroundPath);
			if(mode == 1){
				foregroundHSV = frameRGBToHSV(foreground);
				// substitute
				// extractAndReplaceGreenScreen(foreground, background, foregroundHSV);
				// substitute and blend edge
				extractAndReplaceGreenScreen(foreground, background, foregroundHSV, linearLength);
				loadByteArrToBuffer(foreground, processedImgBuffer[i]);
			}else{
				if(useHSV){
				// HSV
					foregroundHSV = frameRGBToHSV(foreground);
					if(allowGaussianBlur){
						foregroundHSVBlurred = GaussianBlur(foregroundHSV, sigmaMatrix);
					}
					extractAndReplaceMovingHSV(foreground, background, foregroundHSV, i);
					loadByteArrToBuffer(foreground, processedImgBuffer[i]);
					updateCache(foregroundHSV, i);
				}else{
					// RGB
					foregroundRGB = frameRGBByteToInt(foreground);
					if(allowGrayScale){
						
					}
					if(allowGaussianBlur){
						foregroundRGBBlurred = GaussianBlur(foregroundRGB, sigmaMatrix);
						extractAndReplaceMovingRGB(foreground, background, foregroundRGBBlurred, i);
						loadByteArrToBuffer(foreground, processedImgBuffer[i]);
						updateCache(foregroundRGBBlurred, i);
					}else{
						extractAndReplaceMovingRGB(foreground, background, foregroundRGB, i);
						loadByteArrToBuffer(foreground, processedImgBuffer[i]);
						updateCache(foregroundRGB, i);
					}
					
					// foregroundRGBBlurredByte = frameRGBIntToByte(foregroundRGBBlurred);

				}


				// foreground = frameRGBIntToByte(foregroundRGBAvg);
				// loadByteArrToBuffer(foreground, processedImgBuffer[i]);

				


				// // currentOriginalRGBForeground = foregroundRGB.clone();
				// // extractAndReplaceMovingRGB(foreground, background, foregroundRGB, i);
				// // loadByteArrToBuffer(foreground, processedImgBuffer[i]);
				// // updateCache(currentOriginalRGBForeground, i);
			}

			
		}

		lbIm1List = new JLabel[frameNumber];
		for (int i = 0; i < frameNumber; i++) {
			lbIm1List[i] = new JLabel(new ImageIcon(processedImgBuffer[i]));
		}

		// Use label to display the image
		// set once
		frame = new JFrame();
		GridBagLayout gLayout = new GridBagLayout();
		frame.getContentPane().setLayout(gLayout);

		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.gridx = 0;
		c.gridy = 0;

		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 1;

		long startPlayTimestemp = System.currentTimeMillis();

		double intervalMS = 1000/fps;

		JLabel curLabel = new JLabel();
		curLabel.setIcon(new ImageIcon(processedImgBuffer[0]));
		frame.getContentPane().add(curLabel, c);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		// frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		frame.setVisible(true);

		// if mode 0 set buffer 0 - 478 need to change
		// int frameNumberDisplay = frameNumber;
		// if(mode == 0){
		// 	frameNumberDisplay--;
		// }

		frame.setAlwaysOnTop(true);
		frame.setAlwaysOnTop(false);
		// display
		for (int i = 1; i < frameNumber; i++) {
			try {
				// next time - now
                wait(startPlayTimestemp + (int)intervalMS*(i+1)-System.currentTimeMillis());
            } catch (Exception e) {
				e.printStackTrace();;
			}
			curLabel.setIcon(new ImageIcon(processedImgBuffer[i]));
			frame.repaint();
			System.out.println("freshing " + i);
		}

		
		System.out.println(String.format("use time %sms, %s intervals", String.valueOf(System.currentTimeMillis() - startPlayTimestemp), String.valueOf(frameNumber)));
		
		frame.setVisible(false);
		frame.dispose();
	}

	public static void main(String[] args) {

		// String parentDict = "C:\\Users\\14048\\Desktop\\multimedia\\assignment2\\";
		// args = new String[]{parentDict + "input/foreground_1", parentDict + "input/background_static_2", "1"};		
		// args = new String[]{parentDict + "input/foreground_1", parentDict + "input/background_moving_1", "1"};		
		// args = new String[]{parentDict + "subtraction/background_subtraction_1", parentDict + "input/background_static_2", "0"};
		// args = new String[]{"C:\\Users\\14048\\Desktop\\multimedia\\assignment2\\input\\foreground_3", "C:\\Users\\14048\\Desktop\\multimedia\\assignment2\\input\\background_moving_1", "1"};
		// System.out.println(Arrays.toString(args));
		
		VideoDisplay VD = new VideoDisplay(args);
		// paramter setting, all have default
		// display
		VD.fps = 24;

		// mode 0
		VD.linearLength = 3; // if 1, no linear length
		// always use HSV

		// mode 1
		// how many past frame to check
		VD.pastFrameCacheLength = 8;

		// average background check
		VD.allowAvgExtract = true;// to avoid leaves it must be true
		VD.useHSV = false;
		VD.HSVTolerateRange = new float[]{(float)0.01, (float)0.5, (float)0.5};
		VD.HSVAvgTolerateRange = new float[]{(float)0.2, (float)1, (float)1}; // tolerate a small swift on environment extraction, moving
		VD.RGBTolerateRange = new int[]{5, 5, 5};
		VD.RGBAvgTolerateRange = new int[]{50, 50, 50};

		// allow grayscale
		// VD.allowGrayScale = false;

		// allow gaussian blur
		VD.allowGaussianBlur = true;
		VD.expandSize = 3; // bigger blurrer
		VD.sigma = 0.5; // smaller blurrer

		// // naive
		// VD.pastFrameCacheLength = 1;
		// VD.allowAvgExtract = false;
		// VD.allowGaussianBlur = false;

		VD.run();
		System.exit(0);
	}

}
