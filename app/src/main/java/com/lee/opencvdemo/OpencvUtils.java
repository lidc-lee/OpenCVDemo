package com.lee.opencvdemo;


import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.floodFill;
import static org.opencv.imgproc.Imgproc.resize;


/**
 * Created by AA on 2016/11/14.
 * 身份证识别关键技术
 */

public class OpencvUtils {
    private static final String TAG = "OpencvUtils";

    private int[] char_result;

    /**
     * 识别身份证
     * @param inputImage
     */
    public  void recognizeIDCardInfo(Mat inputImage,String path){
//        获取通道图片
        Mat imgChannel = getImageChannel(inputImage);
        //获取身份证号码区域
        List<RotatedRect> rects = positionDetect(imgChannel);
        Log.e(TAG,rects.size()+"");
        //获取身份证号码字符矩阵
        Mat output = normalPosArea(imgChannel,rects.get(0));
        Log.e(TAG,"获取身份证号码字符矩阵"+output.size());
        //获得切割的矩阵
        List<Mat> char_mat = char_segmet(output);

        ANN_MLP ann_mlp = ANN_MLP.create();
        try {
            ann_train(ann_mlp,10,24,path);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        classify(ann_mlp,char_mat);

    }

    private void classify(ANN_MLP ann_mlp, List<Mat> char_mat) {
        char_result = new int[char_mat.size()];
        for (int i=0;i<char_mat.size();i++){
//            Log.e(TAG,"char_mat.get(i)"+char_mat.get(i));
            Mat output = new Mat(1,10,CvType.CV_32FC1);
            Mat char_feature = calcGradientFeat(char_mat.get(i));
            ann_mlp.predict(char_feature,output,0);
            Core.MinMaxLocResult result = Core.minMaxLoc(output);
            char_result[i] = (int) result.maxLoc.x;
            Log.e(TAG,"char_result"+char_result[i]);
        }
    }

    private Mat calcGradientFeat(Mat input){
        List<Float> feat = new ArrayList<>();
        Mat imag = new Mat();
        Imgproc.resize(input,imag,new Size(8,16));
        //计算x方向和y方向的滤波
        float mask[][] = {{1f/8f,2f/8f,1f/8f},{0,0,0},{-1f/8f,-2f/8f,-1f/8f}};
        Mat y_mask = new Mat(3,3,CvType.CV_32F);
        for (int i =0;i<y_mask.rows();i++){
            for (int j =0;j<y_mask.cols();j++){
                y_mask.put(i,j,mask[i][j]);
//                Log.e(TAG,"y_mask"+y_mask.get(i,j)[0]);
            }
        }
        //转置
        Mat x_mask = new Mat();
        Core.transpose(y_mask,x_mask);
        Mat sobelX = new Mat();
        Mat sobelY = new Mat();
        Imgproc.filter2D(imag,sobelX,CvType.CV_32F,x_mask);
        Imgproc.filter2D(imag,sobelY,CvType.CV_32F,y_mask);
        Core.convertScaleAbs(sobelX,sobelX);
        Core.convertScaleAbs(sobelY,sobelY);//取绝对值

        float totleValueX = sumMatValue(sobelX);
        float totleValueY = sumMatValue(sobelY);
        Log.e(TAG,"totleValueX--" + totleValueX + "\ntotleValuey--"+totleValueY);

        for (int i = 0;i<imag.rows();i =i+4){
            for (int j = 0;j<imag.cols();j =j+4){
                Mat subImageX = new Mat(sobelX,new Rect(j,i,4,4));
                feat.add(sumMatValue(subImageX)/totleValueX);
                Mat subImageY = new Mat(sobelY,new Rect(j,i,4,4));
                feat.add(sumMatValue(subImageY)/totleValueY);
            }
        }

        //计算第2个特征
        Mat gray = new Mat();
        Imgproc.resize(input,gray,new Size(4,8));
        Mat p = gray.reshape(1,1);
        p.convertTo(p,CvType.CV_32FC1);
        for (int i =0;i<p.cols();i++){
            feat.add((float)p.get(0,i)[0]);
        }

        //增加水平直方图
        Mat vhist = projectHistogram(input,1);//水平
        Mat hhist = projectHistogram(input,0);//垂直
        for (int i = 0;i<vhist.cols();i++){
            feat.add((float) vhist.get(0,i)[0]);
        }
        for (int j = 0;j<hhist.cols();j++){
            feat.add((float) hhist.get(0,j)[0]);
        }
        Mat dis = Mat.zeros(1,feat.size(),CvType.CV_32F);
        for (int i =0;i<feat.size();i++){
            dis.put(0,i,feat.get(i));
        }
        return dis;
    }
    private Mat projectHistogram(Mat intput,int t){
        Mat lowData = new Mat();
        Imgproc.resize(intput,lowData,new Size(8,16));
        int sz = (t !=0 ) ? lowData.rows() : lowData.cols();
        Mat mhist = Mat.zeros(1,sz,CvType.CV_32F);
//        Log.e(TAG,"lowData"+lowData+"--sz--"+sz);

        for (int j=0;j<sz;j++){
            Mat data = (t != 0) ? lowData.row(j) : lowData.col(j);
            mhist.put(0,j,Core.countNonZero(data));
        }
        Core.MinMaxLocResult minMaxLocResul = Core.minMaxLoc(mhist);
        double min = minMaxLocResul.minVal;
        double max = minMaxLocResul.maxVal;
        if (max>0){
            mhist.convertTo(mhist,-1,1.0f/max,0);
        }
        return mhist;
    }
    private float sumMatValue(Mat intput){
        float sumValue = 0;
        int r = intput.rows();
        int c = intput.cols();
//        if (intput.isContinuous()){ //判断是否连续，如连续可当一维数组
//            c = r*c;
//            r = 1;
//        }
//        Log.e(TAG,"sumMatValue--"+intput.toString()+"---"+c+"--"+r);
        for (int i =0;i<r;i++){
            for (int j =0;j<c;j++){
                sumValue += intput.get(i,j)[0];
            }
        }
        return sumValue;
    }

    /**
     * 获取R通道，输入图像的行列数位450*600
     * @param inputImage
     * @return
     */
    private Mat getImageChannel(Mat inputImage){
        //分割BGR3通道
        List<Mat> splitRGB = new ArrayList<>(inputImage.channels());
        Core.split(inputImage,splitRGB);
        if (inputImage.cols()>600){
            Mat mat = new Mat(450,600, CvType.CV_8UC1);
            resize(splitRGB.get(2),mat,mat.size());
            return mat;
        }else {
            return splitRGB.get(2);
        }
    }

    /**
     * 获取身份证号码的区域
     * @param inputImage
     */
    private List<RotatedRect> positionDetect(Mat inputImage){
        List<RotatedRect> rects = new ArrayList<>();
        //二值化输出Mat
         Mat threshold_R = OstuBeresenThreshold(inputImage);

        Mat imaInv = new Mat(inputImage.size(),inputImage.type(),Scalar.all(255));
        //黑白色翻转，即背景色为黑色
        Mat threshold_Inv = new Mat();
        Core.subtract(imaInv,threshold_R,threshold_Inv);
//        Log.e(TAG,"--黑白色翻转--"+threshold_Inv.toString());
        //闭形态学结构元素
        Size size = new Size(15,3);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,size);
        Imgproc.morphologyEx(threshold_Inv,threshold_Inv,Imgproc.CV_MOP_CLOSE,element);
//        Log.e(TAG,"--morphologyEx--"+threshold_Inv.toString()+"\n--element--"+element.toString());

        //只检测外轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(threshold_Inv,contours,imaInv,Imgproc.CV_RETR_EXTERNAL,Imgproc.CV_CHAIN_APPROX_NONE);
//        Log.e(TAG,"--imaInv--"+imaInv);
//        Log.e(TAG,"--contours.size--"+contours.size());



//        //对候选的轮廓进一步筛选
        for (int i =0;i<contours.size();i++){
            Point[] points = contours.get(i).toArray();
//            Log.e(TAG,"--contours--"+points[0]);
            MatOfPoint2f point2f = new MatOfPoint2f(points);
            RotatedRect rotatedRect = Imgproc.minAreaRect(point2f);
//            Log.e(TAG,"--RotatedRect--size.width--"+rotatedRect.size.width+"--size.height--"
//                    +rotatedRect.size.height);
            if (!isEligible(rotatedRect)){
                contours.remove(i);//删除
            }else {
                rects.add(rotatedRect);//插入
            }
        }
        //测试是否找到了号码区域
//        Mat outputimg = new Mat();
//        inputImage.copyTo(outputimg);

        return rects;
    }

    /**
     * 二值化
     * @param inputImage
     * @return
     */
    private Mat OstuBeresenThreshold(Mat inputImage){
        Mat output = new Mat();
        double otsu_T = Imgproc.threshold(inputImage,output,0,255,Imgproc.CV_THRESH_OTSU);
        double min;
        double max;
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(inputImage);
        min = minMaxLocResult.minVal;
        max = minMaxLocResult.maxVal;

        final double CT = 0.12;
        double beta = CT * (max-min+1)/128;
        double beta_lowT = (1-beta) * otsu_T;
        double beta_highT = (1+ beta) * otsu_T;
        Mat doubleMatIn = new Mat();
        inputImage.copyTo(doubleMatIn);
        int rows = doubleMatIn.rows();
        int cols = doubleMatIn.cols();

        double Tbn;
        for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
                double tmp[] = doubleMatIn.get(i,j);
                double tmp2[] = output.get(i,j);
                if (i<2 || i>rows-3 || j<2 || j>rows-3){
                    if (tmp[0] < beta_lowT){
                        tmp2[0] = 0;
                    }else {
                        tmp2[0] = 255;
                    }
                }else {
                    Rect rect = new Rect(i-2,j-2,5,5);
                    Mat tt = doubleMatIn.submat(rect);
                    Scalar scalar = Core.sumElems(tt);
                    Tbn = scalar.val[0]/25;
                    if (tmp[0]<beta_lowT || tmp[0]<Tbn && (beta_lowT<=tmp[0] && tmp[0]>=beta_highT)){
                        tmp2[0] = 0;
                    }else {
                        tmp2[0] = 255;
                    }
                }
                output.put(i,j,tmp2);
//                Log.e(TAG,output.get(i,j)[0]+"");
            }

        }
        Log.e(TAG,"--OstuBeresenThreshold--"+output.size());
        return output;
    }

    private boolean isEligible(RotatedRect rect){
        float error = 0.4f;
        final float aspect = (float)(4.5/0.3);//长宽比
        int min = (int) (10 * aspect * 10);//最小区域
        int max = (int) (50 * aspect * 50);//最大区域
        float rmin = aspect - 2*aspect*error;//考虑误差后的最小长宽比
        float rmax = aspect + 2*aspect*error;//考虑误差后的最大长宽比

        int area = (int)rect.size.height * (int)rect.size.width;
        float r = (float) (rect.size.width / rect.size.height);

        if (r<1){
            r = 1/r;
        }
//        Log.e(TAG,"--min--"+min+"--max--"+max+"--rmin--"+rmin+"--rmax--"+rmax);
//        Log.e(TAG,"--area--"+area+ "--r--"+r);
        if ((area<min|| area>max) || (r<rmin || r>rmax)) {
            //满足该条件该区域为号码区域
            return false;
        }else{
            return true;
        }

    }

    private Mat normalPosArea(Mat inputImage,RotatedRect rect){
        Mat output = new Mat();

        float r;
        float angle;
        angle = (float) rect.angle;
        r = (float) (rect.size.width/rect.size.height);

        if (r<1){
            angle = 90 + angle;
        }
        Mat rotmat = Imgproc.getRotationMatrix2D(rect.center,angle,1);//获取变型矩阵
        Mat img_rotated = new Mat();
        Imgproc.warpAffine(inputImage,img_rotated,rotmat,inputImage.size(),Imgproc.CV_INTER_CUBIC);
        //裁剪图像
        Size rect_size = rect.size;
        if (r<1){
            double tmp = rect_size.width;
            rect_size.width = rect_size.height;
            rect_size.height = tmp;
        }
        Mat img_crop = new Mat();
        Imgproc.getRectSubPix(img_rotated,rect_size,rect.center,img_crop);
        //用关照直观图调整所有裁剪得到的图像，使得相同宽度和高度，适用于训练和分类
        Mat resultResized = new Mat();
        resultResized.create(20,300,CvType.CV_8UC1);
        resize(img_crop,resultResized,resultResized.size(),0,0,Imgproc.INTER_CUBIC);
        resultResized.copyTo(output);

        return output;
    }

    /**
     * 获取切割得到的字符矩阵
     * @param inputImg
     * @return
     */
    private List<Mat> char_segmet(Mat inputImg){

        Mat img_threshold = new Mat();
        Mat whiteImg = new Mat(inputImg.size(),inputImg.type(),new Scalar(255));
        Mat in_Inv = new Mat();
        Core.subtract(whiteImg,inputImg,in_Inv);
        //大律法二值化
        Imgproc.threshold(in_Inv,img_threshold,0,255,Imgproc.CV_THRESH_OTSU);
        Log.e(TAG,"threshold"+img_threshold.toString());

        int x_char[] = new int[19];
        short counter = 1;
        short num = 0;

        boolean flag[] =new boolean[img_threshold.cols()];
        Log.e(TAG,"flag"+flag.length);
        for (int j =0;j<img_threshold.cols();j++){
            flag[j] =true;
            for (int i =0; i<img_threshold.rows();i++){
//                Log.e(TAG,"img_threshold.get(i,j)"+img_threshold.get(i,j)[0]);
                if (img_threshold.get(i,j)[0]!=0){
                    flag[j] = false;
                    break;
                }
            }
        }
        for (int i =0;i<19;i++){
            Log.e(TAG,"x_char["+i+"]="+x_char[i]);
        }

        for (int i=0; i<img_threshold.cols()-2;i++){
            if (flag[i]){
                x_char[counter] += i;
                num++;
                if (!flag[i+1]&& !flag[i+2]){
                    x_char[counter] = x_char[counter]/num;
                    num = 0;
                    counter++;
                }
            }
//            Log.e(TAG,"x_char["+counter+"]="+x_char[counter]);

        }

        List<Mat> output = new ArrayList<>();
        x_char[18] = img_threshold.cols();
        for (int i =0; i<18; i++){
            Rect rect = new Rect(x_char[i],0,Math.abs(x_char[i+1]-x_char[i]),img_threshold.rows());
            Mat mat = new Mat(in_Inv,rect);
            output.add(mat);

        }
        for (int i =0;i<19;i++){
            Log.e(TAG,"x_char["+i+"]="+x_char[i]);
        }
        return output;
    }

    private ANN_MLP ann_train(ANN_MLP ann_mlp, int numCharacters,int nlayers,String pathJson) throws JSONException {
        List<Mat> list = new ArrayList<>();
        list = ann_json(pathJson);
        Mat trainData = list.get(0);
        Mat classes = list.get(1);

        Mat layerSizes = new Mat(1,3,CvType.CV_32SC1);//3层神经网络
        layerSizes.put(0,0,trainData.cols());   //  输入层的神经元节点数设置位24
        layerSizes.put(0,1,nlayers);            //  1个隐藏层的神经元节点数，设置位24
        layerSizes.put(0,2,numCharacters);      //  输出层的神经元节点数为10

        ann_mlp.setLayerSizes(layerSizes);
        ann_mlp.setActivationFunction(ANN_MLP.SIGMOID_SYM,1,1);
        ann_mlp.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER+TermCriteria.EPS,5000,0.01));

        Mat trainClasses = new Mat();
        trainClasses.create(trainData.rows(),numCharacters,CvType.CV_32FC1);

        for (int i = 0;i<trainData.rows();i++){
            for (int j = 0;j<trainClasses.cols();j++){
                if (j == (int)classes.get(0,j)[0]){
                    trainClasses.put(i,j,1);
                }else {
                    trainClasses.put(i,j,0);
                }
            }
        }

        ann_mlp.train(trainData, Ml.ROW_SAMPLE,trainClasses);
        return ann_mlp;

    }

    //解析json数据
    public  List<Mat> ann_json(String jsonPath) throws JSONException {
        JSONObject jsonObject = new JSONObject(jsonPath);
        JSONObject trainingData = jsonObject.getJSONObject("TrainingData");
        JSONObject classes = jsonObject.getJSONObject("classes");
        Mat trainMat = fzMat(trainingData);
        Mat classMat = fzMat(classes);
        Log.e(TAG,trainMat.toString()+"--"+trainMat.size());
        Log.e(TAG,classMat.toString()+"--" + classMat.size());
        List<Mat> list = new ArrayList<>();
        list.add(trainMat);
        list.add(classMat);
        return list;
    }

    private  Mat fzMat(JSONObject data) throws JSONException {
        
        int dt = 0;
        List<String> dataStr = new ArrayList<>();
        String strData = data.getString("data");
        int rows = data.getInt("rows");
        int cols = data.getInt("cols");
        if (data.getString("dt").equals("f")){
            dt = CvType.CV_32F;
        }else if (data.getString("dt").equals("u")){
            dt = CvType.CV_8UC1;
        }

        int start = 0;
        for (int k=0;k<strData.length();k++){
            if (strData.charAt(k)==' '){
                StringBuffer stringBuffer = new StringBuffer();
                stringBuffer.append(strData,start,k-1);
                dataStr.add(stringBuffer.toString());
//                Log.e(TAG,stringBuffer.toString());
                start = k+1;
            }else if (k == strData.length()-1){
                StringBuffer stringBuffer = new StringBuffer();
                stringBuffer.append(strData,start,k);
                dataStr.add(stringBuffer.toString());
            }

        }
        Log.e(TAG,rows+"---"+cols+"---"+dataStr.size());
        Mat mat = new Mat( rows,cols,dt);

        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (dt == CvType.CV_8UC1){
                    mat.put(i,j,dataStr.get(j).getBytes());
                }else if (dt == CvType.CV_32F){
                    mat.put(i,j,Float.valueOf(dataStr.get(j)));
                }
            }
        }
        return mat;

    }

}
