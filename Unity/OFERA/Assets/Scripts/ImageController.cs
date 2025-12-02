using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;


[DisallowMultipleComponent]
public class ImageController : MonoBehaviour
{
    [Header("Targets")]
    [Tooltip("이미지를 표시할 RawImage (UI).")]
    public RawImage targetImage;

    [Header("UI")]
    [Tooltip("캡션 텍스트 (TextMeshProUGUI). 비워두면 아래 Text를 사용하거나 생략됩니다.")]
    public TextMeshProUGUI captionTMP;
    [Tooltip("남은 시간을 표시할 TextMeshProUGUI (선택).")]
    public TextMeshProUGUI timeTMP;

    [Header("Playback Flags (Inspector controllable)")]
    [Tooltip("true면 리스트 전체를 순차적으로 재생")]
    public bool continuousPlay = true;
    [Tooltip("continuousPlay=false 일 때만 의미: true면 현재 클립 반복 재생")]
    public bool loop = false;
    [Tooltip("버튼(또는 외부 호출)로 다음 클립으로 이동시키는 트리거. Update에서 소비됩니다.")]
    public bool nextClip = false;
    [Tooltip("true면 nextClip은 무시되고, 각 이미지의 duration 이후 자동으로 다음으로 이동")]
    public bool timeLimitClip = false;

    [Header("List of Images")]
    public List<ImageItem> items = new List<ImageItem>();

    [Header("State")]
    public int currentIndex = 0;
    [Tooltip("시작 시 자동 재생")]
    public bool playOnStart = true;

    float _timer = 0f;
    bool _isPlaying = false;

    [Serializable]
    public struct ImageItem
    {
        [Tooltip("표시할 텍스처 (PNG/JPG 등).")]
        public Texture2D texture;
        [Tooltip("캡션 텍스트 (비어 있으면 texture.name 사용).")]
        public string text;
        [Tooltip("timeLimitClip=true일 때 이 이미지가 표시될 시간(초).")]
        public float duration;

        public ImageItem(Texture2D tex, string t = "", float d = 3f)
        {
            texture = tex;
            text = t;
            duration = Mathf.Max(0f, d);
        }
    }

    void Awake()
    {
        if (targetImage == null)
        {
            targetImage = GetComponentInChildren<RawImage>();
        }
        // 캡션 컴포넌트 자동 탐색(있으면)

        if (captionTMP == null) captionTMP = GetComponentInChildren<TextMeshProUGUI>();
    }

    void Start()
    {
        if (items.Count == 0)
        {
            Debug.LogWarning("[ImageController] items가 비어 있습니다.");
            return;
        }

        currentIndex = Mathf.Clamp(currentIndex, 0, items.Count - 1);
        ApplyItem(currentIndex);

        _isPlaying = playOnStart;
        _timer = 0f;
    }

    void Update()
    {
        if (items.Count == 0 || targetImage == null) { UpdateTimeUI(0f, 0f, false); return; }

        // 테스트용 키(선택): N=다음, L=루프 토글, C=연속 토글
        if (Input.GetKeyDown(KeyCode.N)) nextClip = true;
        if (Input.GetKeyDown(KeyCode.L)) loop = !loop;
        if (Input.GetKeyDown(KeyCode.C)) continuousPlay = !continuousPlay;

        if (!_isPlaying)
        {
            // 재생 중이 아니면 숨기거나 0초 표시
            UpdateTimeUI(0f, 0f, false);
            return;
        }

        var item = items[currentIndex];

        if (timeLimitClip)
        {
            _timer += Time.deltaTime;
            float dur = Mathf.Max(0.01f, item.duration <= 0f ? 3f : item.duration);

            // ▼ 남은 시간 UI 업데이트
            UpdateTimeUI(_timer, dur, true);

            if (_timer >= dur)
            {
                _timer = 0f;
                GoNext_Auto();
            }
            nextClip = false; // timeLimitClip 중에는 무시
        }
        else
        {
            // timeLimitClip이 꺼져 있으면 남은 시간 UI 비활성/초기화
            UpdateTimeUI(0f, 0f, false);

            if (nextClip)
            {
                nextClip = false;
                GoNext_Manual();
            }
        }
    }

    // ====== Public Controls ======

    public void Play()
    {
        _isPlaying = true;
        _timer = 0f;
        targetImage.gameObject.SetActive(true);
        captionTMP.gameObject.SetActive(true);
        timeTMP.gameObject.SetActive(true);
    }

    public void Pause()
    {
        _isPlaying = false;
    }

    public void Stop()
    {
        _isPlaying = false;
        _timer = 0f;
        // currentIndex = 0;
        ApplyItem(currentIndex);
        targetImage.gameObject.SetActive(false);
        captionTMP.gameObject.SetActive(false);
        timeTMP.gameObject.SetActive(false);
    }

    public void SetIndex(int idx)
    {
        if (items.Count == 0) return;
        currentIndex = Mathf.Clamp(idx, 0, items.Count - 1);
        _timer = 0f;
        ApplyItem(currentIndex);
    }

    public void SetImageList(List<ImageItem> newItems, bool autoPlay = true, int startIndex = 0)
    {
        items = newItems ?? new List<ImageItem>();
        if (items.Count == 0) return;

        currentIndex = Mathf.Clamp(startIndex, 0, items.Count - 1);
        ApplyItem(currentIndex);
        _timer = 0f;
        _isPlaying = autoPlay;
    }

    // ====== Internal Transitions ======

    void GoNext_Auto()
    {
        if (continuousPlay)
        {
            // 전체 순차 재생
            int next = currentIndex + 1;
            if (next >= items.Count)
            {
                // 리스트 끝
                _isPlaying = false; // 자동 종료
                return;
            }
            currentIndex = next;
            ApplyItem(currentIndex);
        }
        else
        {
            // continuousPlay=false 인 경우 자동으로는 넘어가지 않음
            if (loop)
            {
                // 현재 이미지 반복(다시 시작)
                _timer = 0f;
                ApplyItem(currentIndex);
            }
            else
            {
                // 멈춤
                _isPlaying = false;
            }
        }
    }

    void GoNext_Manual()
    {
        if (continuousPlay)
        {
            // 전체 순차 재생 중 "다음" 입력 시
            int next = currentIndex + 1;
            if (next >= items.Count)
            {
                // 끝이면 멈춤
                _isPlaying = false;
                return;
            }
            currentIndex = next;
            _timer = 0f;
            ApplyItem(currentIndex);
        }
        else
        {
            // 개별 모드: loop / stop / nextClip
            if (loop)
            {
                // 현재 이미지 반복 (다음 명령이 와도 계속 같은 이미지)
                _timer = 0f;
                ApplyItem(currentIndex);
            }
            else
            {
                // 수동으로만 다음으로
                int next = currentIndex + 1;
                if (next >= items.Count)
                {
                    // 더 이상 없으면 멈춤
                    _isPlaying = false;
                    return;
                }
                currentIndex = next;
                _timer = 0f;
                ApplyItem(currentIndex);
            }
        }
    }

    // ====== Rendering ======

    void ApplyItem(int index)
    {
        if (items.Count == 0 || targetImage == null) return;

        index = Mathf.Clamp(index, 0, items.Count - 1);
        var item = items[index];

        // 이미지 표시
        if (item.texture != null)
        {
            targetImage.texture = item.texture;
            // RawImage의 비율을 텍스처에 맞게 조정하고 싶다면 여기서 조정 가능(선택):
            // targetImage.SetNativeSize();
        }
        else
        {
            targetImage.texture = null;
        }

        // 캡션 텍스트
        string label = !string.IsNullOrEmpty(item.text)
            ? item.text
            : (item.texture != null ? item.texture.name : string.Empty);


        if (captionTMP != null) captionTMP.text = label;

        // 새 항목으로 바뀔 때 타이머 초기화 & UI 갱신
        _timer = 0f;
        if (timeLimitClip)
        {
            float dur = Mathf.Max(0.01f, items[index].duration <= 0f ? 3f : items[index].duration);
            UpdateTimeUI(0f, dur, true);
        }
        else
        {
            UpdateTimeUI(0f, 0f, false);
        }
    }

    void UpdateTimeUI(float elapsed, float duration, bool show)
    {
        if (timeTMP == null) return;

        if (!show || duration <= 0f)
        {
            timeTMP.text = "";
            return;
        }
        float remain = Mathf.Max(0f, duration - elapsed);
        timeTMP.text = remain.ToString("0.00"); // 소수 둘째자리
    }
}
