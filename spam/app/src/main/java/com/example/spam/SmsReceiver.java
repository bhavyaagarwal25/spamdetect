package com.example.spam;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.telephony.SmsMessage;
import android.provider.Telephony;
import android.util.Log;

import androidx.core.app.NotificationCompat;

public class SmsReceiver extends BroadcastReceiver {

    private static final String TAG = "SmsReceiver";
    private static final String CHANNEL_ID = "spam_channel_id";
    private static final int NOTIFICATION_ID = 1;

    @Override
    public void onReceive(Context context, Intent intent) {
        Log.d(TAG, "SMS received");
        
        if (Telephony.Sms.Intents.SMS_RECEIVED_ACTION.equals(intent.getAction())) {
            Bundle bundle = intent.getExtras();

            if (bundle != null) {
                Object[] pdus = (Object[]) bundle.get("pdus");
                if (pdus != null) {
                    String format = bundle.getString("format");
                    StringBuilder fullMessage = new StringBuilder();
                    String sender = null;

                    for (Object pdu : pdus) {
                        SmsMessage sms = SmsMessage.createFromPdu((byte[]) pdu, format);
                        sender = sms.getDisplayOriginatingAddress();
                        fullMessage.append(sms.getMessageBody());
                    }

                    String messageText = fullMessage.toString().toLowerCase();
                    if (isSpamMessage(messageText)) {
                        Log.d(TAG, "Spam detected from: " + sender);
                        showSpamNotification(context, sender, messageText);
                    }
                }
            }
        }
    }

    private boolean isSpamMessage(String message) {
        // Common spam keywords and patterns
        String[] spamKeywords = {
            "win", "winner", "prize", "congratulation", "lottery", "offer",
            "free", "cash", "money", "click", "link", "urgent", "account",
            "bank", "verify", "limited time", "claim", "reward"
        };

        message = message.toLowerCase();
        for (String keyword : spamKeywords) {
            if (message.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private void showSpamNotification(Context context, String sender, String message) {
        NotificationManager notificationManager =
                (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);

        if (notificationManager == null) {
            Log.e(TAG, "NotificationManager is null");
            return;
        }

        // Create notification channel for Android O+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            CharSequence name = "Spam Detection Channel";
            String description = "Notifications for detected spam messages";
            int importance = NotificationManager.IMPORTANCE_HIGH;

            NotificationChannel channel = new NotificationChannel(CHANNEL_ID, name, importance);
            channel.setDescription(description);
            channel.enableVibration(true);
            channel.setVibrationPattern(new long[]{100, 200, 300, 400, 500});
            channel.setShowBadge(true);

            notificationManager.createNotificationChannel(channel);
        }

        // Create intent to open app when notification is tapped
        Intent intent = new Intent(context, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        intent.putExtra("spam_sender", sender);
        intent.putExtra("spam_message", message);

        PendingIntent pendingIntent = PendingIntent.getActivity(
                context,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        // Build the notification
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("⚠️ Spam Message Detected!")
                .setContentText("From: " + sender)
                .setStyle(new NotificationCompat.BigTextStyle()
                        .bigText("From: " + sender + "\n" + message))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_MESSAGE)
                .setAutoCancel(true)
                .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
                .setContentIntent(pendingIntent)
                .setVibrate(new long[]{100, 200, 300, 400, 500});

        // Show the notification
        notificationManager.notify(NOTIFICATION_ID, builder.build());
        Log.d(TAG, "Notification shown for spam from: " + sender);
    }

    // Method to test the notification
    public static void testNotification(Context context) {
        Intent testIntent = new Intent("com.example.spam.TEST_NOTIFICATION");
        new SmsReceiver().onReceive(context, testIntent);
    }
}
