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

public class IDActivity extends Activity {
    Button bt_sc;
    ImageView iv_sfz;
    TextView tv_id_card;
    Button bt_sb;
    boolean flag=false;

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

        bt_sc.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                chooseImgFromCamera();
            }
        });
        bt_sb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                if (flag){
                    if (!OpenCVLoader.initDebug()){
                        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0,IDActivity.this,baseLoaderCallback);
                    }else {
                        baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//                    }
                }
            }
        });

    }
    /**打开照相机拍照*/
    private void chooseImgFromCamera() {
        Intent intentFromCamera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        if (hasSdcard()){
            intentFromCamera.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(new File(Environment.getExternalStorageDirectory(), "idcard.jpg")));
        }

        startActivityForResult(intentFromCamera, 121);
    }

    private static boolean hasSdcard() {
        String state = Environment.getExternalStorageState();

        if (state.equals(Environment.MEDIA_MOUNTED)){
            return true;
        }
        else{
            return false;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode){
            case 121:
                if (hasSdcard()){
                    File tempFile = new File(Environment.getExternalStorageDirectory(), "idcard.jpg");

                    BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
                    bmpFactoryOptions.inJustDecodeBounds = true;
                    bmpFactoryOptions.inSampleSize = calculateInSampleSize(bmpFactoryOptions, 1280, 800);

                    bmpFactoryOptions.inJustDecodeBounds = false;

                    Bitmap photo = BitmapFactory.decodeFile(tempFile.getPath(), bmpFactoryOptions);
                    iv_sfz.setImageBitmap(photo);

                }
                else{
                    Toast.makeText(this,"没有SD卡!",Toast.LENGTH_LONG);
                }
                break;
            case 122:
                if (data != null){
                    setImageToHeadView(data);
                }
                break;
        }
        super.onActivityResult(requestCode, resultCode, data);
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

    private void cropRawPhoto(Uri uri){
        Intent intent = new Intent("com.android.camera.action.CROP");
        intent.setDataAndType(uri, "image/*");

        intent.putExtra("crop", "true");

        intent.putExtra("aspectX", 1.5);//宽高比例
        intent.putExtra("aspectY", 1);

        intent.putExtra("outputX", 300);
        intent.putExtra("outputY", 200);
        intent.putExtra("return-data", true);

        startActivityForResult(intent, 122);
    }
    private void setImageToHeadView(Intent intent){
        Bundle extras = intent.getExtras();

        if (extras != null){
            Bitmap photo = extras.getParcelable("data");
            iv_sfz.setImageBitmap(photo);
            flag = true;
        }else {
            flag = false;
        }
    }

    private BaseLoaderCallback baseLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS){

                String imageFilePath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/idcard.jpg";

//                String imagePath = Environment.getExternalStorageDirectory().getAbsolutePath()+"/sfz.jpg";
//                Log.e("imagePath",imagePath);
                Mat matPath = Imgcodecs.imread(imageFilePath);
                OpencvUtils opencvUtils = new OpencvUtils();
                opencvUtils.recognizeIDCardInfo(matPath,ReadJson());
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
