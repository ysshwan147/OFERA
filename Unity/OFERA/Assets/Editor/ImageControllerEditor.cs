#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(ImageController))]
public class ImageControllerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        // 기존 필드들 기본 표시
        DrawDefaultInspector();

        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("Quick Actions", EditorStyles.boldLabel);

        var c = (ImageController)target;

        // Start / Stop
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Start"))
        {
            c.Play();
            EditorUtility.SetDirty(c);
        }
        if (GUILayout.Button("Stop"))
        {
            c.Stop();
            EditorUtility.SetDirty(c);
        }
        EditorGUILayout.EndHorizontal();

        // Previous / Next
        // 요청하신 동작 규칙:
        // - continuousPlay == false 일 때만 Previous/Next 사용
        // - timeLimitClip == false 일 때만 Next 사용(= Update에서 nextClip 트리거를 소비)
        bool itemsReady = c != null && c.items != null && c.items.Count > 0;
        bool disablePrevNextBase = !itemsReady || c.continuousPlay;
        bool disablePrev = disablePrevNextBase || c.currentIndex <= 0;
        bool disableNext = disablePrevNextBase || c.timeLimitClip || c.currentIndex >= (itemsReady ? c.items.Count - 1 : 0);

        EditorGUILayout.BeginHorizontal();

        using (new EditorGUI.DisabledScope(disablePrev))
        {
            if (GUILayout.Button("Previous"))
            {
                // 단일 모드(prev): editor 모드에서도 즉시 인덱스 이동
                c.SetIndex(Mathf.Clamp(c.currentIndex - 1, 0, c.items.Count - 1));
                EditorUtility.SetDirty(c);
            }
        }

        using (new EditorGUI.DisabledScope(disableNext))
        {
            if (GUILayout.Button("Next"))
            {
                if (Application.isPlaying)
                {
                    // 단일 모드 + timeLimitClip=false 일 때만 동작
                    // Update()에서 nextClip 트리거를 소비해서 넘깁니다.
                    c.nextClip = true;
                }
                else
                {
                    // 에디터 정지 상태에서도 보기 좋게 다음으로 이동
                    c.SetIndex(Mathf.Clamp(c.currentIndex + 1, 0, c.items.Count - 1));
                }
                EditorUtility.SetDirty(c);
            }
        }

        EditorGUILayout.EndHorizontal();

        // 실행 중이면 인스펙터 갱신(버튼 활성/비활성 반영)
        if (Application.isPlaying)
            Repaint();
    }
}
#endif
