package com.lee.opencvdemo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import static org.opencv.imgcodecs.Imgcodecs.imread;

/**
 * Created by AA on 2016/11/15.
 */

public class IDActivity extends Activity implements View.OnClickListener {
    private static final int CAMERA_RESULT = 100;
    Button bt_sc;
    ImageView iv_sfz;
    TextView tv_id_card;
    Button bt_sb;
    private String imageFilePath;
    private boolean isPictureSelected=false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.id_card_activity);
        initView();

    }

    private void initView(){
        bt_sc = (Button) findViewById(R.id.bt_sc);
        iv_sfz = (ImageView) findViewById(R.id.iv_sf);
        tv_id_card = (TextView) findViewById(R.id.tv_id_card);
        bt_sb = (Button) findViewById(R.id.bt_sb);

        bt_sc.setOnClickListener(this);
        bt_sb.setOnClickListener(this);

    }
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.bt_sc:
                openCamera();
                break;
            case R.id.bt_sb:
                if (!OpenCVLoader.initDebug()) {
                    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, IDActivity.this, baseLoaderCallback);
                } else {
                    baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
                }
                break;
        }
    }
    private void openCamera() {
        imageFilePath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/idcard.jpg";
        File imageFile = new File(imageFilePath);
        Uri imageFileUri = Uri.fromFile(imageFile);

        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        i.putExtra(MediaStore.EXTRA_OUTPUT, imageFileUri);
        startActivityForResult(i, CAMERA_RESULT);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {

            BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
            bmpFactoryOptions.inJustDecodeBounds = true;

            bmpFactoryOptions.inSampleSize = calculateInSampleSize(bmpFactoryOptions, 1280, 800);

            bmpFactoryOptions.inJustDecodeBounds = false;

            Bitmap bmp = BitmapFactory.decodeFile(imageFilePath, bmpFactoryOptions);

            iv_sfz.setImageBitmap(bmp);

//            saveBitmap(bmp);

            isPictureSelected = true;
        }
    }
    private int calculateInSampleSize(BitmapFactory.Options options,
                                      int reqWidth, int reqHeight) {

        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {

            final int heightRatio = Math.round((float) height
                    / (float) reqHeight);
            final int widthRatio = Math.round((float) width / (float) reqWidth);

            inSampleSize = heightRatio < widthRatio ? widthRatio : heightRatio;
        }

        return inSampleSize;
    }


    private BaseLoaderCallback baseLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS){

//                if (!isPictureSelected) {
//                    Toast.makeText(IDActivity.this, "请先选取图片", Toast.LENGTH_LONG);
//                    return;
//                }
                imageFilePath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/idcard.jpg";
                Mat matPath = Imgcodecs.imread(imageFilePath);
                OpencvUtils opencvUtils = new OpencvUtils();
                opencvUtils.recognizeIDCardInfo(matPath,ReadJson());
                Log.e("imageFilePath",imageFilePath);

//                String result = OpenCVHelper.recognizeIDCard(getAssets(), "ann_xml.xml", imageFilePath);
//                String[] ret = result.split(",");
//                tv_id_card.setText(ret[1]);
            }else {
                super.onManagerConnected(status);
            }
        }
    };

    private String ReadJson(){
        String jsonSource = "";
        try {
            InputStream inputStream = this.getAssets().open("ann.json");
            //读取文件流
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream,"utf-8");
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String temp = "";
            while ((temp = bufferedReader.readLine())!= null){
                jsonSource += temp;
            }
            bufferedReader.close();
            inputStream.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return jsonSource;
    }
}
