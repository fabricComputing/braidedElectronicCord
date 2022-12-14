package com.epic.localmusicnoserver.adapter;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.support.v7.app.AlertDialog;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.epic.localmusicnoserver.R;
import com.epic.localmusicnoserver.activity.HomeActivity;
import com.epic.localmusicnoserver.activity.PlaylistActivity;
import com.epic.localmusicnoserver.database.DBManager;
import com.epic.localmusicnoserver.entity.PlayListInfo;
import com.mcxtzhang.swipemenulib.SwipeMenuLayout;

import java.util.ArrayList;
import java.util.List;


public class HomeListViewAdapter extends BaseAdapter {

    private List<PlayListInfo> dataList;

    private Activity activity;

    private DBManager dbManager;


    public HomeListViewAdapter(List<PlayListInfo> dataList, Activity activity, DBManager dbManager) {
        this.dataList = dataList;
        this.activity = activity;
        this.dbManager = dbManager;
    }


    @Override
    public int getCount() {
        if (dataList.size() == 0)
            return 1;
        else
            return dataList.size();
    }


    @Override
    public Object getItem(int position) {
        return dataList.get(position);
    }


    @Override
    public long getItemId(int position) {
        return position;
    }


    @Override
    public View getView(final int position, View convertView, ViewGroup parent) {
        final Holder holder;
        if (convertView == null){
            holder = new Holder();
            convertView = LayoutInflater.from(activity).inflate(R.layout.play_list_view_item,null,false);
            holder.swipView =  (View)convertView.findViewById(R.id.play_list_content_swip_view);
            holder.contentLayout = (LinearLayout) convertView.findViewById(R.id.play_list_content_ll);
            holder.coverImage = (ImageView) convertView.findViewById(R.id.play_list_cover_iv);
            holder.listName = (TextView) convertView.findViewById(R.id.play_list_name_tv);
            holder.listCount = (TextView) convertView.findViewById(R.id.play_list_music_count_tv);
            holder.deleteButton = (Button) convertView.findViewById(R.id.playlist_swip_delete_menu_btn);
            convertView.setTag(holder);
        }else {
            holder = (Holder)convertView.getTag();
        }

        if (dataList.size() == 0){
            //?????????????????????????????????
            holder.listName.setText("????????????");
            holder.listName.setGravity(Gravity.CENTER_VERTICAL);
            holder.listCount.setVisibility(View.GONE);
            ((SwipeMenuLayout)holder.swipView).setSwipeEnable(false);
        }else {
            //???????????????????????????
            PlayListInfo playListInfo = dataList.get(position);
            holder.listName.setText(playListInfo.getName());
            holder.listCount.setText(playListInfo.getCount()+"???");
            holder.listName.setGravity(Gravity.BOTTOM);
            holder.listCount.setVisibility(View.VISIBLE);
            ((SwipeMenuLayout)holder.swipView).setSwipeEnable(true);
        }

        holder.contentLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (dataList.size() == 0){
                    //????????????
                    final AlertDialog.Builder builder = new AlertDialog.Builder(activity);
                    View view = LayoutInflater.from(activity).inflate(R.layout.dialog_create_playlist,null);
                    final EditText playlistEt = (EditText)view.findViewById(R.id.dialog_playlist_name_et);
                    builder.setView(view);
                    builder.setPositiveButton("??????", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            String name = playlistEt.getText().toString();
                            dbManager.createPlaylist(name);
                            updateDataList();
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
                }else {  //?????????????????????????????????
                    Intent intent = new Intent(activity, PlaylistActivity.class);  //????????????
                    intent.putExtra("playlistInfo",dataList.get(position));
                    activity.startActivity(intent);
                }
            }
        });

        holder.deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                View view = LayoutInflater.from(activity).inflate(R.layout.dialog_delete_playlist,null);
                final AlertDialog.Builder builder = new AlertDialog.Builder(activity);
                builder.setView(view);
                builder.setPositiveButton("??????", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dbManager.deletePlaylist(dataList.get(position).getId());
                        ((SwipeMenuLayout) holder.swipView).quickClose();
                        dialog.dismiss();
                        updateDataList();
                    }
                });

                builder.setNegativeButton("??????", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        ((SwipeMenuLayout) holder.swipView).quickClose();
                        dialog.dismiss();
                    }
                });
                builder.show();
            }
        });
        return convertView;
    }


    /**
     * ???????????????????????????
     */
    public void updateDataList(){
        List<PlayListInfo> playListInfoList = new ArrayList<>();
        playListInfoList = dbManager.getMyPlayList();
        dataList.clear();
        dataList.addAll(playListInfoList);
        notifyDataSetChanged();
        ((HomeActivity)activity).updatePlaylistCount();
    }

    /**
     * ?????????Holder
     */
    class Holder{
        View swipView;
        LinearLayout contentLayout;
        ImageView coverImage;
        TextView listName;
        TextView listCount;
        Button deleteButton;
    }
}
