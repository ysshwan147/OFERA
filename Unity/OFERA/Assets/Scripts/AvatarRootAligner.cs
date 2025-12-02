using System.Collections;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[DisallowMultipleComponent]
public class AvatarRootAligner : MonoBehaviour
{
    [Header("References")]
    [Tooltip("VR 리그의 카메라(센터 아이) 또는 일반 메인 카메라. 비워두면 자동 탐색합니다.")]
    public Transform cameraTransform;

    [Header("Placement")]
    [Tooltip("카메라 정면으로부터 떨어뜨릴 거리 (m)")]
    public float forwardDistance = 1.5f;
    [Tooltip("배치 시 추가 높이 오프셋 (m)")]
    public float heightOffset = 0.0f;
    [Tooltip("Yaw(수평 회전)만 맞추기 (권장)")]
    public bool alignYawOnly = true;
    [Tooltip("바닥에 스냅(Physics.Raycast로 바닥 감지)")]
    public bool snapToFloor = false;
    [Tooltip("바닥 레이어 마스크")]
    public LayerMask floorMask = ~0;

    // ─────────────────────────────────────────────────────────────
    // 인스펙터 버튼/컨텍스트 메뉴로 호출할 공개 함수
    // ─────────────────────────────────────────────────────────────
    [ContextMenu("Align Now")]
    public void AlignNow()
    {
        // 카메라 자동 탐색
        if (cameraTransform == null)
        {
            var cam = Camera.main;
            if (cam != null) cameraTransform = cam.transform;
        }

        if (cameraTransform == null)
        {
            Debug.LogWarning("[AvatarRootAligner] cameraTransform을 찾지 못해 정렬을 건너뜁니다.");
            return;
        }

        // 카메라의 수평 방향 벡터 계산
        Vector3 fwd = cameraTransform.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
        fwd.Normalize();

        // 타겟 위치: 카메라 앞 + 높이 오프셋
        Vector3 targetPos = cameraTransform.position + fwd * forwardDistance;
        targetPos.y += heightOffset;

        // 바닥 스냅(선택)
        if (snapToFloor)
        {
            Ray ray = new Ray(targetPos + Vector3.up * 2f, Vector3.down);
            if (Physics.Raycast(ray, out RaycastHit hit, 10f, floorMask, QueryTriggerInteraction.Ignore))
            {
                targetPos = hit.point;
            }
        }

        // 회전: yaw만 또는 전체 회전
        Quaternion targetRot;
        if (alignYawOnly)
        {
            Vector3 camFwd = cameraTransform.forward;
            camFwd.y = 0f;
            if (camFwd.sqrMagnitude < 1e-6f) camFwd = Vector3.forward;
            camFwd = -camFwd;
            targetRot = Quaternion.LookRotation(camFwd, Vector3.up);
        }
        else
        {
            targetRot = cameraTransform.rotation * Quaternion.Euler(0, 180f, 0);
        }

        // 배치
        transform.SetPositionAndRotation(targetPos, targetRot);
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(AvatarRootAligner))]
public class AvatarRootAlignerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var aligner = (AvatarRootAligner)target;

        EditorGUILayout.Space();
        if (GUILayout.Button("Align Now"))
        {
            Undo.RecordObject(aligner.transform, "Align Avatar Root");
            aligner.AlignNow();
            EditorUtility.SetDirty(aligner.transform);
        }
    }
}
#endif
