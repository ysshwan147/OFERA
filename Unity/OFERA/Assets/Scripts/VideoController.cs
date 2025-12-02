using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using TMPro;

[System.Serializable]
public class ClipEntry
{
    public VideoClip clip;             // 재생할 비디오
    [TextArea] public string text;     // 표시할 텍스트 (비워두면 clip.name 사용)
    public bool useTimeLimit = false;  // 이 클립에서 시간 제한을 사용할지
    [Min(0f)] public float timeLimitSeconds = 3f; // 시간 제한(초). 0이면 clip.length 사용
}

public class VideoController : MonoBehaviour
{
    [Header("Inspector: Data")]
    public List<ClipEntry> entries = new List<ClipEntry>();
    public VideoPlayer videoPlayer;
    public TMP_Text clipNameText;
    public TextMeshProUGUI timeTMP;

    [Header("Inspector: Controls (토글/트리거)")]
    public bool continuousPlay = true;     // true: 전체 순차 재생
    public bool loop = false;              // continuousPlay=false 일 때만 동작. true면 현재 클립 반복
    public bool timeLimitMode = false;     // true면 nextClip 트리거 무시, 각 클립의 시간 지나면 자동 다음
    public bool nextClipTrigger = false;   // continuousPlay=false && timeLimitMode=false 일 때만 동작 (Inspector에서 true로 켰다 끄는 트리거)

    [Header("State (읽기전용)")]
    [SerializeField] private int currentIndex = 0;
    [SerializeField] private bool isSequenceRunning = false;
    [SerializeField] private bool isClipRunning = false;

    // 내부 상태
    private Coroutine playSequenceCo;
    private Coroutine playSingleCo;

    // ▼ 남은 시간 UI/제어용
    [SerializeField] private float _timer = 0f;             // 현재 클립에서 경과 시간(초)
    [SerializeField] private float _currentDuration = 0f;   // 현재 클립의 제한 시간(초). 0이면 표시/카운트 안 함

    void Awake()
    {
        if (videoPlayer == null)
        {
            videoPlayer = GetComponent<VideoPlayer>();
        }

        // 권장 세팅
        videoPlayer.playOnAwake = false;
        videoPlayer.waitForFirstFrame = true;
        videoPlayer.skipOnDrop = false;
        videoPlayer.audioOutputMode = VideoAudioOutputMode.Direct;

        videoPlayer.errorReceived += (vp, msg) =>
        {
            Debug.LogError($"[VideoPlayer] Error: {msg}");
        };
    }

    void Update()
    {
        // nextClip 트리거 처리 (한 프레임 동안 true였다면 동작 후 false로 되돌림)
        if (nextClipTrigger)
        {
            // 조건: continuousPlay=false && timeLimitMode=false 일 때만 동작
            if (!continuousPlay && !timeLimitMode)
            {
                NextClip();
            }
            nextClipTrigger = false; // 트리거 reset
        }

        // 루프 모드 동기화 (continuousPlay=false 일 때만 적용)
        if (!continuousPlay)
        {
            videoPlayer.isLooping = loop;
        }
        else
        {
            // 순차 재생 중에는 per-clip loop는 끔
            videoPlayer.isLooping = false;
        }

        // ▼ 남은 시간 UI/타이머 갱신
        bool showTimer = false;
        if (timeLimitMode && isClipRunning && entries.Count > 0)
        {
            var entry = entries[Mathf.Clamp(currentIndex, 0, entries.Count - 1)];
            if (entry.useTimeLimit && _currentDuration > 0f && videoPlayer.isPlaying)
            {
                _timer += Time.deltaTime;
                showTimer = true;
            }
        }
        UpdateTimeUI(showTimer);
    }

    /* === 외부에서 호출 가능한 편의 메서드 === */

    public void StartPlayback()
    {
        StopAllPlayback();

        videoPlayer.gameObject.SetActive(true);
        clipNameText.gameObject.SetActive(true);

        if (timeTMP != null)
        {
            timeTMP.gameObject.SetActive(true);
        }

        if (entries == null || entries.Count == 0)
        {
            Debug.LogWarning("[VideoController] Entries가 비었습니다.");
            UpdateTimeUI(false); // UI 초기화
            return;
        }

        if (continuousPlay)
        {
            playSequenceCo = StartCoroutine(PlayAllInSequenceFrom(currentIndex));
        }
        else
        {
            playSingleCo = StartCoroutine(PlayCurrentSingleMode());
        }
    }

    public void StopAllPlayback()
    {
        if (playSequenceCo != null) StopCoroutine(playSequenceCo);
        if (playSingleCo != null) StopCoroutine(playSingleCo);
        playSequenceCo = null;
        playSingleCo = null;

        isSequenceRunning = false;
        isClipRunning = false;

        if (videoPlayer != null)
        {
            videoPlayer.Stop();
        }

        _timer = 0f;
        _currentDuration = 0f;
        UpdateTimeUI(false);

        videoPlayer.gameObject.SetActive(false);
        clipNameText.gameObject.SetActive(false);
        if (timeTMP != null)
        {
            timeTMP.gameObject.SetActive(false);
        }
    }

    public void NextClip()
    {
        if (continuousPlay) return;      // continuous 모드에선 수동 넘김 X
        if (timeLimitMode) return;       // 타임리밋 모드에선 수동 넘김 X

        // 다음 인덱스로 이동
        int next = currentIndex + 1;
        if (next >= entries.Count)
        {
            Debug.Log("[VideoController] 마지막 클립 이후입니다.");
            return;
        }

        currentIndex = next;

        // 현재 재생 중이면 갈아타기
        if (isClipRunning)
        {
            // 한 클립만 재생하는 코루틴을 재시작
            if (playSingleCo != null) StopCoroutine(playSingleCo);
            playSingleCo = StartCoroutine(PlayCurrentSingleMode());
        }
        else
        {
            // 재생 중이 아니면 로드 후 대기
            PrepareAndPlay(entries[currentIndex]).Run(this);
        }
    }

    public void PreviousClip()
    {
        if (continuousPlay) return;
        if (timeLimitMode) return;

        int prev = currentIndex - 1;
        if (prev < 0)
        {
            Debug.Log("[VideoController] 첫 번째 클립 이전입니다.");
            return;
        }

        currentIndex = prev;

        if (isClipRunning)
        {
            if (playSingleCo != null) StopCoroutine(playSingleCo);
            playSingleCo = StartCoroutine(PlayCurrentSingleMode());
        }
        else
        {
            PrepareAndPlay(entries[currentIndex]).Run(this);
        }
    }

    /* === 내부 코루틴 / 헬퍼 === */

    private IEnumerator PlayAllInSequenceFrom(int startIndex)
    {
        isSequenceRunning = true;

        for (int i = Mathf.Max(0, startIndex); i < entries.Count; i++)
        {
            currentIndex = i;
            var entry = entries[i];

            // 루프 금지(순차 재생에서는 항상 한 번만)
            videoPlayer.isLooping = false;

            // 준비 및 시작
            yield return PrepareAndPlay(entry);

            // 타임리밋 모드면 해당 시간 후 자동 다음
            if (timeLimitMode && entry.useTimeLimit)
            {
                double limit = GetEffectiveLimitSeconds(entry);
                yield return WaitTimeLimitOrStop(limit);
            }
            else
            {
                // 타임리밋 모드가 아니면 정상 종료까지 기다림
                yield return WaitUntilClipEnds();
            }
        }

        isSequenceRunning = false;
        isClipRunning = false;

        // 끝났으니 UI 정리
        _currentDuration = 0f;
        _timer = 0f;
        UpdateTimeUI(false);
    }

    private IEnumerator PlayCurrentSingleMode()
    {
        isClipRunning = true;
        var entry = entries[Mathf.Clamp(currentIndex, 0, entries.Count - 1)];

        // single 모드에서 loop 토글 반영
        videoPlayer.isLooping = loop;

        // 준비 및 시작
        yield return PrepareAndPlay(entry);

        if (timeLimitMode && entry.useTimeLimit)
        {
            double limit = GetEffectiveLimitSeconds(entry);
            yield return WaitTimeLimitOrStop(limit);

            // 타임리밋으로 넘어간 경우, 다음 클립으로 자동 이동
            int next = currentIndex + 1;
            if (next < entries.Count)
            {
                currentIndex = next;
                if (playSingleCo != null) StopCoroutine(playSingleCo);
                playSingleCo = StartCoroutine(PlayCurrentSingleMode());
                yield break;
            }
            else
            {
                // 마지막이면 종료
                isClipRunning = false;
                _currentDuration = 0f;
                _timer = 0f;
                UpdateTimeUI(false);
                yield break;
            }
        }
        else
        {
            // 타임리밋 모드가 아니면: 루프면 그대로 유지(코루틴 종료), 루프 아니면 끝날 때까지 기다렸다가 멈춤
            if (!loop)
            {
                yield return WaitUntilClipEnds();
                isClipRunning = false;
                _currentDuration = 0f;
                _timer = 0f;
                UpdateTimeUI(false);
            }
            else
            {
                // loop=true면 코루틴은 종료하지만 VideoPlayer는 계속 루프 재생
                isClipRunning = false;
                // 루프에서는 제한시간 UI 사용하지 않음
                _currentDuration = 0f;
                _timer = 0f;
                UpdateTimeUI(false);
            }
        }
    }

    private IEnumerator PrepareAndPlay(ClipEntry entry)
    {
        // 텍스트 설정 (비어있으면 clip.name)
        if (clipNameText != null)
        {
            string label = (!string.IsNullOrEmpty(entry.text)) ? entry.text :
                           (entry.clip != null ? entry.clip.name : "(No Clip)");
            clipNameText.text = label;
        }

        // 안전 정리
        videoPlayer.Stop();
        videoPlayer.clip = entry.clip;
        videoPlayer.time = 0;
        videoPlayer.frame = 0;

        // 준비
        videoPlayer.Prepare();
        yield return new WaitUntil(() => videoPlayer.isPrepared);

        // 재생
        videoPlayer.Play();
        yield return new WaitUntil(() => videoPlayer.isPlaying);

        // ▼ 타임리밋 초기화 (이 클립이 시작된 시점)
        _timer = 0f;
        _currentDuration = (timeLimitMode && entry.useTimeLimit)
            ? (float)GetEffectiveLimitSeconds(entry)
            : 0f;
        UpdateTimeUI(_currentDuration > 0f);
    }

    private IEnumerator WaitUntilClipEnds()
    {
        bool finished = false;
        VideoPlayer.EventHandler handler = null;
        handler = (VideoPlayer vp) => { finished = true; };

        videoPlayer.loopPointReached += handler;

        // 안전 타임아웃(클립 길이 + 2초)
        double timeout = Time.timeAsDouble + (videoPlayer.clip != null ? videoPlayer.clip.length : 10.0) + 2.0;

        while (!finished && Time.timeAsDouble < timeout)
        {
            yield return null;
        }

        videoPlayer.loopPointReached -= handler;
    }

    private IEnumerator WaitTimeLimitOrStop(double seconds)
    {
        float target = Mathf.Max(0f, (float)seconds);

        // _timer가 target에 도달할 때까지 대기 (Update에서 _timer 증가)
        while (_timer < target)
        {
            yield return null;
        }

        // 타임리밋 도달 → 다음 클립으로 넘어갈 준비를 위해 정지
        videoPlayer.Stop();

        // UI 초기화(다음 클립에서 다시 세팅)
        _currentDuration = 0f;
        _timer = 0f;
        UpdateTimeUI(false);
    }

    private double GetEffectiveLimitSeconds(ClipEntry entry)
    {
        // timeLimitSeconds <= 0이면 clip.length 사용 (없으면 기본 3초)
        if (entry.timeLimitSeconds > 0f) return entry.timeLimitSeconds;
        if (entry.clip != null && entry.clip.length > 0.0) return entry.clip.length;
        return 3.0;
    }

    /* === 디버그/도움 메서드 === */

    // 현재 항목을 다시 로드
    public void ReloadCurrent()
    {
        if (entries.Count == 0) return;

        if (continuousPlay)
        {
            StartPlayback();
        }
        else
        {
            if (playSingleCo != null) StopCoroutine(playSingleCo);
            playSingleCo = StartCoroutine(PlayCurrentSingleMode());
        }
    }

    // 인덱스로 이동 (single 모드에서 유용)
    public void JumpTo(int index)
    {
        if (index < 0 || index >= entries.Count) return;
        currentIndex = index;

        if (!continuousPlay)
        {
            if (playSingleCo != null) StopCoroutine(playSingleCo);
            playSingleCo = StartCoroutine(PlayCurrentSingleMode());
        }
        else
        {
            // 순차모드면 해당 인덱스부터 다시 순차 재생
            StartPlayback();
        }
    }

    // ===== 남은 시간 UI 갱신 =====
    private void UpdateTimeUI(bool show)
    {
        if (timeTMP == null)
            return;

        if (!show || _currentDuration <= 0f)
        {
            timeTMP.text = "";
            return;
        }

        float remain = Mathf.Max(0f, _currentDuration - _timer);
        timeTMP.text = remain.ToString("0.00"); // 소수 둘째자리
    }
}

// 코루틴 호출 보조 (Optional)
static class CoroutineExt
{
    public static void Run(this IEnumerator co, MonoBehaviour host)
    {
        if (co != null) host.StartCoroutine(co);
    }
}
