import java.awt.*;
import java.awt.image.*;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.io.*;
import java.lang.reflect.Array;
import java.util.Arrays;

import javax.swing.*;
// import java.awt.Color;
import javax.swing.text.StyledEditorKit.BoldAction;

import java.lang.Math;

// import java.lang.*;
// import org.opencv.imgproc.Imgproc;
public class RGB2PNG {

	JFrame frame;
	JLabel[] lbIm1List;
	BufferedImage[] processedImgBuffer;
	int width; // default image width and height
	int height;
	int frameNumber;

	int layerLength;
	int layerLength2;
	int frameLength;

	String imgPathComponent1;
	public String[] pngOutputPath;
	// C:\Users\14048\Desktop\multimedia\project\video_rgb\SAL_490_270_437\SAL_490_270_437.

	int mode;

	public RGB2PNG(String[] args) {
		try {
			// folder path at [0], rgb ver
			String[] splitted1 = args[0].split("/|\\\\");
			// for (String item : splitted1) {
			// System.out.println(item);
			// }
			String[] splitted2 = splitted1[splitted1.length - 1].split("_");
			width = Integer.parseInt(splitted2[1]);
			height = Integer.parseInt(splitted2[2]);
			frameNumber = Integer.parseInt(splitted2[3]);

			// init outputPath
			pngOutputPath = new String[frameNumber];

			layerLength = height * width;
			layerLength2 = layerLength * 2;
			frameLength = layerLength * 3;

			imgPathComponent1 = args[0];
			if (imgPathComponent1.charAt(imgPathComponent1.length() - 1) + "" != "/"
					| imgPathComponent1.charAt(imgPathComponent1.length() - 1) + "" != "\\") {
				imgPathComponent1 += "/";
			}
			imgPathComponent1 += splitted1[splitted1.length - 1] + ".";

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Read Image RGB
	 * Reads the image of given width and height at the given imgPath into the
	 * provided BufferedImage.
	 */
	private byte[] readAnImageRGB(int width, int height, String imgPath) {
		try {
			File file = new File(imgPath);
			RandomAccessFile raf = new RandomAccessFile(file, "r");
			raf.seek(0);
			byte[] bytes = new byte[frameLength];
			raf.read(bytes);
			bytes = rearrangeRGB(bytes);
			return bytes;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	private byte[] rearrangeRGB(byte[] bytes) {
		// RGBRGBRGB... to RRR.GGG.BBB.
		byte[] rearranged = new byte[frameLength];
		for (int i = 0; i < layerLength; i++) {
			int oldR = 3 * i;
			rearranged[i] = bytes[oldR];
			rearranged[i + layerLength] = bytes[oldR + 1];
			rearranged[i + layerLength2] = bytes[oldR + 2];
		}
		return rearranged;
	}

	private void loadByteArrToBuffer(byte[] bytes, BufferedImage img) {

		int ind = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				byte a = 0;
				byte r = bytes[ind];
				byte g = bytes[ind + layerLength];
				byte b = bytes[ind + layerLength2];

				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
				// int pix = ((a << 24) + (r << 16) + (g << 8) + b);
				img.setRGB(x, y, pix);
				ind++;
			}
		}

	}

	private int[] frameRGBByteToInt(byte[] RGB) {
		// H ∈ [0°, 360°], saturation S ∈ [0, 1], and value V ∈ [0, 1]
		// H ∈ [0, 1]
		int[] res = new int[frameLength];
		int ind = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int posInLayer1 = ind;
				int posInLayer2 = ind + layerLength;
				int posInLayer3 = ind + layerLength2;
				res[posInLayer1] = RGB[posInLayer1] & 0xff;
				res[posInLayer2] = RGB[posInLayer2] & 0xff;
				res[posInLayer3] = RGB[posInLayer3] & 0xff;
				ind++;
			}
		}
		return res;
	}

	private byte[] frameRGBIntToByte(int[] RGB) {
		// RGB must in int [0,255]
		byte[] res = new byte[frameLength];
		for (int i = 0; i < frameLength; i++) {
			res[i] = (byte) RGB[i];
		}
		return res;
	}

	private double[] spaceConversionByteToDouble(double[][] conversionMatrix, byte[] rgbBytes) {
		double[] res = new double[layerLength * 3];
		for (int i = 0; i < layerLength; i++) {
			byte r = rgbBytes[i];
			byte g = rgbBytes[i + layerLength];
			byte b = rgbBytes[i + layerLength * 2];
			double[] temp = calculateMatrixWithoutBound(conversionMatrix, r, g, b);
			res[i] = temp[0];
			res[i + layerLength] = temp[1];
			res[i + layerLength * 2] = temp[2];
		}
		return res;
	}

	private byte[] spaceConversionDoubleToByte(double[][] conversionMatrix, double[] rgbBytes) {
		byte[] res = new byte[layerLength * 3];
		for (int i = 0; i < layerLength; i++) {
			double r = rgbBytes[i];
			double g = rgbBytes[i + layerLength];
			double b = rgbBytes[i + layerLength * 2];
			byte[] temp = calculateMatrixWithBound(conversionMatrix, r, g, b);
			res[i] = temp[0];
			res[i + layerLength] = temp[1];
			res[i + layerLength * 2] = temp[2];
		}
		return res;
	}

	private double[] calculateMatrixWithoutBound(double[][] matrix, byte a, byte b, byte c) {
		// input abc -128-127 to 0-255 cal-> double
		int a0 = a & 0xff;
		int b0 = b & 0xff;
		int c0 = c & 0xff;
		double x = Math.round(matrix[0][0] * a0 + matrix[0][1] * b0 + matrix[0][2] * c0);
		double y = Math.round(matrix[1][0] * a0 + matrix[1][1] * b0 + matrix[1][2] * c0);
		double z = Math.round(matrix[2][0] * a0 + matrix[2][1] * b0 + matrix[2][2] * c0);
		return new double[] { x, y, z };
	}

	private byte[] calculateMatrixWithBound(double[][] matrix, double a, double b, double c) {
		// input double to 0-255 bounded, to bytes
		double x = Math.round(matrix[0][0] * a + matrix[0][1] * b + matrix[0][2] * c);
		double y = Math.round(matrix[1][0] * a + matrix[1][1] * b + matrix[1][2] * c);
		double z = Math.round(matrix[2][0] * a + matrix[2][1] * b + matrix[2][2] * c);
		int lowBound = 0;
		int highBound = 255;
		if (x < lowBound) {
			x = lowBound;
		} else if (x > highBound) {
			x = highBound;
		}
		if (y < lowBound) {
			y = lowBound;
		} else if (y > highBound) {
			y = highBound;
		}
		if (z < lowBound) {
			z = lowBound;
		} else if (z > highBound) {
			z = highBound;
		}
		return new byte[] { (byte) x, (byte) y, (byte) z };
	}

	private double[][] rgb2yuv = {
			{ 0.299, 0.587, 0.114 },
			{ -0.147, -0.289, 0.436 },
			{ 0.615, -0.515, -0.1 }
	};
	private double[][] yuv2rgb = {
			{ 1.000, 0, 1.140 },
			{ 1.000, -0.395, -0.581 },
			{ 1.000, 2.032, 0 }
	};

	private double[][] rgb2yiq = {
			{ 0.299, 0.587, 0.114 },
			{ 0.596, -0.274, -0.322 },
			{ 0.211, -0.523, 0.312 }
	};
	private double[][] yiq2rgb = {
			{ 1.000, 0.956, 0.621 },
			{ 1.000, -0.272, -0.647 },
			{ 1.000, -1.106, 1.703 }
	};

	int fps = 30;

	public synchronized void run() {
		// // Read a parameter from command line
		// String param1 = args[1];
		// System.out.println("The second parameter was: " + param1);

		// initialize buffers
		processedImgBuffer = new BufferedImage[frameNumber];

		String actualFilePath;
		String outputFilePath;

		byte[] currentFrame = new byte[frameLength];
		try {
			BufferedWriter pngPathsFile = new BufferedWriter(new FileWriter("pngPaths.txt"));
			for (int i = 0; i < frameNumber; i++) {
				// [001, frameNumber]
				actualFilePath = String.format("%s%03d.rgb", imgPathComponent1, i + 1);
				outputFilePath = String.format("%s%03d.png", imgPathComponent1, i + 1);
				// System.out.println(outputFilePath);
				// just initialize buffer
				processedImgBuffer[i] = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
				// all inplace
				currentFrame = readAnImageRGB(width, height, actualFilePath);

				loadByteArrToBuffer(currentFrame, processedImgBuffer[i]);

				// save png file by pmz
				ImageIO.write(processedImgBuffer[i], "png", new File(outputFilePath));
				pngOutputPath[i] = outputFilePath;
				pngPathsFile.write(outputFilePath + '\n');
			}
			pngPathsFile.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// OUTPUT THE PATH OF PNG

		/*
		 * lbIm1List = new JLabel[frameNumber];
		 * for (int i = 0; i < frameNumber; i++) {
		 * lbIm1List[i] = new JLabel(new ImageIcon(processedImgBuffer[i]));
		 * }
		 * 
		 * // Use label to display the image
		 * // set once
		 * frame = new JFrame();
		 * GridBagLayout gLayout = new GridBagLayout();
		 * frame.getContentPane().setLayout(gLayout);
		 * 
		 * GridBagConstraints c = new GridBagConstraints();
		 * c.fill = GridBagConstraints.HORIZONTAL;
		 * c.anchor = GridBagConstraints.CENTER;
		 * c.weightx = 0.5;
		 * c.gridx = 0;
		 * c.gridy = 0;
		 * 
		 * c.fill = GridBagConstraints.HORIZONTAL;
		 * c.gridx = 0;
		 * c.gridy = 1;
		 * 
		 * JLabel curLabel = new JLabel();
		 * curLabel.setIcon(new ImageIcon(processedImgBuffer[0]));
		 * frame.getContentPane().add(curLabel, c);
		 * frame.pack();
		 * frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		 * // frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		 * 
		 * frame.setVisible(true);
		 * 
		 * frame.setAlwaysOnTop(true);
		 * frame.setAlwaysOnTop(false);
		 * 
		 * long startPlayTimestemp = System.currentTimeMillis();
		 * double intervalMS = 1000 / fps;
		 * // display
		 * for (int i = 1; i < frameNumber; i++) {
		 * try {
		 * // next time - now
		 * wait(startPlayTimestemp + (int) intervalMS * (i + 1) -
		 * System.currentTimeMillis());
		 * } catch (Exception e) {
		 * e.printStackTrace();
		 * }
		 * curLabel.setIcon(new ImageIcon(processedImgBuffer[i]));
		 * frame.repaint();
		 * // System.out.println("freshing " + i);
		 * }
		 * 
		 * // System.out.println(String.format("use time %sms, %s intervals",
		 * // String.valueOf(System.currentTimeMillis() - startPlayTimestemp),
		 * // String.valueOf(frameNumber)));
		 * 
		 * frame.setVisible(false);
		 * frame.dispose();
		 */
	}

	public static void main(String[] args) {
		// args[0] parentDict
		String parentDict = "";
		String folder = "";
		// String parentDict =
		// "/Users/piaomz/Downloads/remake-video-project-main/video_rgb/";
		// String parentDict = "C:/Users/14048/Desktop/multimedia/project/video_rgb/";
		// args[1] folder
		if (args.length == 1) {
			parentDict = args[0];
			// folder = args[1];
		} else if (args.length == 2) {
			parentDict = args[0];
			folder = args[1];
		} else {
			System.out.println("missing parameters");
		}

		// String folder = "SAL_490_270_437";
		// String folder = "Stairs_490_270_346";
		// String folder = "video1_240_424_518";
		// String folder = "video2_240_424_383";
		// String folder = "video3_240_424_193";
		String newCameraPath = "";// content [frame, x, y]
		args = new String[] { parentDict + folder, newCameraPath };
		// System.out.println(Arrays.toString(args));
		// for (String item : args) {
		// System.out.println(item);
		// }

		System.out.println("processing, please wait");

		RGB2PNG VD = new RGB2PNG(args);

		// display
		VD.fps = 30;

		VD.run();
		System.exit(0);

	}

}
