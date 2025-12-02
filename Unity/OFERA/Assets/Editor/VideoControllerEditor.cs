#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(VideoController))]
public class VideoControllerEditor : Editor
{
    SerializedProperty entriesProp;
    SerializedProperty videoPlayerProp;
    SerializedProperty clipNameTextProp;
    SerializedProperty timeTMPProp;              // ← 추가: 남은 시간 TMP

    SerializedProperty continuousPlayProp;
    SerializedProperty loopProp;
    SerializedProperty timeLimitModeProp;
    SerializedProperty nextClipTriggerProp;

    SerializedProperty currentIndexProp;

    // ← 추가: 런타임 표시용(읽기 전용)
    SerializedProperty timerProp;               // _timer
    SerializedProperty currentDurationProp;     // _currentDuration

    void OnEnable()
    {
        entriesProp = serializedObject.FindProperty("entries");
        videoPlayerProp = serializedObject.FindProperty("videoPlayer");
        clipNameTextProp = serializedObject.FindProperty("clipNameText");
        timeTMPProp = serializedObject.FindProperty("timeTMP");          // ← 추가

        continuousPlayProp = serializedObject.FindProperty("continuousPlay");
        loopProp = serializedObject.FindProperty("loop");
        timeLimitModeProp = serializedObject.FindProperty("timeLimitMode");
        nextClipTriggerProp = serializedObject.FindProperty("nextClipTrigger");

        currentIndexProp = serializedObject.FindProperty("currentIndex");

        // ← 추가: private serialized 읽기
        timerProp = serializedObject.FindProperty("_timer");
        currentDurationProp = serializedObject.FindProperty("_currentDuration");
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        // Data
        EditorGUILayout.PropertyField(entriesProp, new GUIContent("Entries"), true);
        EditorGUILayout.PropertyField(videoPlayerProp);
        EditorGUILayout.PropertyField(clipNameTextProp);

        // UI
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("UI", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(timeTMPProp, new GUIContent("timeTMP (남은 시간 표시)"));

        // Controls
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);

        EditorGUILayout.PropertyField(continuousPlayProp, new GUIContent("continuousPlay"));
        using (new EditorGUI.DisabledScope(continuousPlayProp.boolValue))
        {
            EditorGUILayout.PropertyField(loopProp, new GUIContent("loop (single mode only)"));
            EditorGUILayout.PropertyField(timeLimitModeProp, new GUIContent("timeLimitMode (single mode only)"));
            EditorGUILayout.PropertyField(nextClipTriggerProp, new GUIContent("nextClipTrigger (toggle)"));
        }

        // State
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("State", EditorStyles.boldLabel);
        using (new EditorGUI.DisabledScope(true))
        {
            EditorGUILayout.PropertyField(currentIndexProp, new GUIContent("currentIndex"));
        }

        // Time Limit Status (read-only)
        DrawTimeLimitStatus();

        // Quick Actions
        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Quick Actions", EditorStyles.boldLabel);

        var controller = (VideoController)target;

        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Start Playback"))
        {
            controller.StartPlayback();
        }
        if (GUILayout.Button("Stop"))
        {
            controller.StopAllPlayback();
        }
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.BeginHorizontal();
        using (new EditorGUI.DisabledScope(controller == null || controller.continuousPlay || controller.timeLimitMode))
        {
            if (GUILayout.Button("Previous"))
            {
                controller.PreviousClip();
            }
            if (GUILayout.Button("Next"))
            {
                controller.NextClip();
            }
        }
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.Space(4);
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Reload Current"))
        {
            controller.ReloadCurrent();
        }

        // JumpTo UI
        int newIndex = Mathf.Clamp(
            EditorGUILayout.IntField(new GUIContent("Jump To Index"), controller != null ? GetCurrentIndex() : 0),
            0,
            GetMaxIndex(controller)
        );
        if (GUILayout.Button("Go", GUILayout.Width(60)))
        {
            controller.JumpTo(newIndex);
        }
        EditorGUILayout.EndHorizontal();

        serializedObject.ApplyModifiedProperties();

        // 인스펙터 갱신(Play중 버튼/남은시간 실시간 반영)
        if (Application.isPlaying)
        {
            Repaint();
        }
    }

    void DrawTimeLimitStatus()
    {
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("Time Limit", EditorStyles.boldLabel);

        using (new EditorGUI.DisabledScope(true))
        {
            float duration = currentDurationProp != null ? currentDurationProp.floatValue : 0f;
            float elapsed = timerProp != null ? timerProp.floatValue : 0f;
            float remain = Mathf.Max(0f, duration - elapsed);

            EditorGUILayout.FloatField("Current Duration (sec)", duration);
            EditorGUILayout.FloatField("Elapsed (sec)", elapsed);
            EditorGUILayout.TextField("Remain (sec)", remain.ToString("0.00"));

            // 작은 진행바(0~1)
            float t = (duration > 0f) ? Mathf.Clamp01(elapsed / duration) : 0f;
            Rect r = GUILayoutUtility.GetRect(18, 18);
            EditorGUI.ProgressBar(r, t, duration > 0f ? $"Progress {Mathf.RoundToInt(t * 100f)}%" : "Inactive");
        }
    }

    int GetCurrentIndex()
    {
        // private serialized field를 사용
        return currentIndexProp != null ? currentIndexProp.intValue : 0;
    }

    int GetMaxIndex(VideoController c)
    {
        if (c == null || c.entries == null) return 0;
        return Mathf.Max(0, c.entries.Count - 1);
    }
}
#endif
