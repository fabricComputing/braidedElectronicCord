package com.epic.localmusicnoserver.view;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.provider.MediaStore;
import android.support.v7.app.AlertDialog;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import com.epic.localmusicnoserver.R;
import com.epic.localmusicnoserver.database.DBManager;
import com.epic.localmusicnoserver.entity.MusicInfo;
import com.epic.localmusicnoserver.entity.PlayListInfo;
import com.epic.localmusicnoserver.service.MusicPlayerService;
import com.epic.localmusicnoserver.util.MusicConstant;
import com.epic.localmusicnoserver.util.MyMusicUtil;

import java.io.File;


public class MusicPopMenuWindow extends PopupWindow{

    private View view;

    private Activity activity;

    private TextView nameText;

    private LinearLayout addLayout;

    private LinearLayout loveLayout;

    private LinearLayout deleteLayout;

    private LinearLayout cancelLayout;

    private MusicInfo musicInfo;

    private PlayListInfo playListInfo;

    private int witchActivity = MusicConstant.ACTIVITY_LOCAL;

    private View parentView;


    public MusicPopMenuWindow(Activity activity, MusicInfo musicInfo,View parentView,int witchActivity) {
        super(activity);
        this.activity = activity;
        this.musicInfo = musicInfo;
        this.parentView = parentView;
        this.witchActivity = witchActivity;
        initView();
    }


   /* public MusicPopMenuWindow(Activity activity, MusicInfo musicInfo,View parentView,int witchActivity,PlayListInfo playListInfo) {
        super(activity);
        this.activity = activity;
        this.musicInfo = musicInfo;
        this.parentView = parentView;
        this.witchActivity = witchActivity;
        this.playListInfo = playListInfo;
        initView();
    }*/


    private void initView(){
        this.view = LayoutInflater.from(activity).inflate(R.layout.pop_window_menu, null);
        this.setContentView(this.view);  // ????????????

        this.setHeight(LinearLayout.LayoutParams.WRAP_CONTENT);  //??????????????????????????????,????????????????????????
        this.setWidth(LinearLayout.LayoutParams.MATCH_PARENT);

        this.setFocusable(true);  //???????????????????????????
        this.setOutsideTouchable(true);  //?????????????????????

        this.setBackgroundDrawable(activity.getResources().getDrawable(R.color.colorWhite));  //???????????????????????????
        this.setAnimationStyle(R.style.pop_window_animation);  //????????????????????????????????????????????????????????????

        this.view.setOnTouchListener(new View.OnTouchListener() {  //??????OnTouchListener???????????????????????????????????????????????????????????????????????????
            public boolean onTouch(View v, MotionEvent event) {
                int height = view.getTop();
                int y = (int) event.getY();
                if (event.getAction() == MotionEvent.ACTION_UP) {
                    if (y < height) {  //?????????????????????
                        dismiss();
                    }
                }
                return true;
            }
        });


        nameText = (TextView)view.findViewById(R.id.popwin_name_tv);
        addLayout = (LinearLayout) view.findViewById(R.id.popwin_add_rl);
        loveLayout = (LinearLayout) view.findViewById(R.id.popwin_love_ll);
        deleteLayout = (LinearLayout) view.findViewById(R.id.popwin_delete_ll);
        cancelLayout = (LinearLayout) view.findViewById(R.id.popwin_cancel_ll);

        nameText.setText("????????? " + this.musicInfo.getName());

        addLayout.setOnClickListener(new View.OnClickListener() {  //?????????
            public void onClick(View v) {
                //????????????????????????????????????
                dismiss();
                AddPlaylistWindow addPlaylistWindow = new AddPlaylistWindow(activity,musicInfo);
                addPlaylistWindow.showAtLocation(parentView, Gravity.BOTTOM|Gravity.CENTER_HORIZONTAL, 0, 0);
                WindowManager.LayoutParams params = activity.getWindow().getAttributes();

                params.alpha=0.7f;  //?????????Popupwindow????????????????????????
                activity.getWindow().setAttributes(params);

                //??????Popupwindow??????????????????Popupwindow?????????????????????1f
                addPlaylistWindow.setOnDismissListener(new PopupWindow.OnDismissListener() {
                    @Override
                    public void onDismiss() {
                        WindowManager.LayoutParams params = activity.getWindow().getAttributes();
                        params.alpha=1f;
                        activity.getWindow().setAttributes(params);
                    }
                });
            }
        });

        loveLayout.setOnClickListener(new View.OnClickListener() {

            public void onClick(View v) {
                MyMusicUtil.setMusicMyLove(activity,musicInfo.getId());
                dismiss();
                View view = LayoutInflater.from(activity).inflate(R.layout.my_love_toast,null);
                Toast toast = new Toast(activity);
                toast.setView(view);
                toast.setDuration(Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER,0,0);
                toast.show();
            }
        });

        deleteLayout.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                deleteOperate(musicInfo,activity);
                dismiss();
            }
        });

        cancelLayout.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                dismiss();
            }
        });
    }

    /**
     * ????????????
     */
    public void deleteOperate(MusicInfo musicInfo,final Context context){
        final int deleteMusicId = musicInfo.getId();
        final int musicId = MyMusicUtil.getIntSharedPreference(MusicConstant.KEY_ID);  //??????????????????Id
        final DBManager dbManager = DBManager.getInstance(context);
        final String path = dbManager.getMusicPath(deleteMusicId);
        LayoutInflater inflater = LayoutInflater.from(context);
        View view = inflater.inflate(R.layout.dialog_delete_file,null);
        final CheckBox deleteFile = (CheckBox)view.findViewById(R.id.dialog_delete_cb);
        final AlertDialog.Builder builder = new AlertDialog.Builder(context);
        AlertDialog dialog = builder.create();

        builder.setView(view);

        builder.setPositiveButton("??????", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (deleteFile.isChecked()){  //????????????????????????
                    File file = new File(path);
                    if (file.exists()) {
                        deleteMediaDB(file,context);
                        dbManager.deleteMusic(deleteMusicId);
                    }
                    if (deleteMusicId == musicId){  //?????????????????????????????????
                        Intent intent = new Intent(MusicPlayerService.PLAYER_MANAGER_ACTION);
                        intent.putExtra(MusicConstant.COMMAND, MusicConstant.COMMAND_STOP);  //??????????????????
                        context.sendBroadcast(intent);
                        MyMusicUtil.setIntSharedPreference(MusicConstant.KEY_ID,dbManager.getFirstId(MusicConstant.LIST_ALLMUSIC));
                    }
                }else {
                    //???????????????
                    if (witchActivity == MusicConstant.ACTIVITY_MYLIST){  //??????
                        dbManager.removeMusicFromPlaylist(deleteMusicId,playListInfo.getId());
                    }else {
                        dbManager.removeMusic(deleteMusicId,witchActivity);
                    }

                    if (deleteMusicId == musicId) {  //?????????????????????????????????
                        Intent intent = new Intent(MusicPlayerService.PLAYER_MANAGER_ACTION);
                        intent.putExtra(MusicConstant.COMMAND, MusicConstant.COMMAND_STOP);
                        context.sendBroadcast(intent);
                    }
                }
                if(onDeleteUpdateListener != null){
                    onDeleteUpdateListener.onDeleteUpdate();
                }
                dialog.dismiss();

            }
        });
        builder.setNegativeButton("??????", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
            }
        });
        builder.show();
    }

    /**
     * ???????????????
     */
    public static void deleteMediaDB(File file,Context context){
        String filePath = file.getPath();
        int res = context.getContentResolver().delete(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                MediaStore.Audio.Media.DATA + "= \"" + filePath+"\"", null);
    }


    private OnDeleteUpdateListener onDeleteUpdateListener;


    public void setOnDeleteUpdateListener(OnDeleteUpdateListener onDeleteUpdateListener){
        this.onDeleteUpdateListener = onDeleteUpdateListener;
    }


    public interface OnDeleteUpdateListener {
        void onDeleteUpdate();
    }
}
