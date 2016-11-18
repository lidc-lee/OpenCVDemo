package com.lee.opencvdemo;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private Button btn_1;
    private Button btn_2;
    private ImageView iv_1;
    private Bitmap image;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn_1 = (Button) findViewById(R.id.bt_1);
        btn_2 = (Button) findViewById(R.id.bt_2);
        iv_1 = (ImageView) findViewById(R.id.iv_1);

        btn_1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent =new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });
        btn_2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!OpenCVLoader.initDebug()){
                    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0,MainActivity.this,baseLoaderCallback);
                }else {
                    baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
                }

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode){
            case 100:
                if (resultCode==RESULT_OK){
                    Uri uri = data.getData();
                    InputStream inputStream =null;
                    try {
                        inputStream =getContentResolver().openInputStream(uri);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                    image = BitmapFactory.decodeStream(inputStream);
                }
                break;
        }
    }

    private BaseLoaderCallback baseLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS){
                mainProcess();
            }else {
                super.onManagerConnected(status);
            }
        }
    };

    private void mainProcess(){
        Mat mat = new Mat();
        Utils.bitmapToMat(this.image,mat);
        Imgproc.putText(mat,"fhgfdsgh",new Point(20,40),3,1,new Scalar(0,255,0,255),2);
        Bitmap bm =Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat,bm);
        iv_1.setImageBitmap(bm);
    }
}
