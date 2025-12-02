using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System;

public class BlendshapeRecorder : MonoBehaviour
{
    public SkinnedMeshRenderer faceMeshRenderer; // Blendshape 데이터를 가지고 있는 렌더러
    private StreamWriter writer;
    private List<string> blendshapeNames = new();
    private bool isRecording = false;

    // 저장 경로 수정 (Assets 폴더 내부 RecordedBlendshapes 사용)
    public string BlendshapeSavePath = "RecordedBlendshapes";
    private string saveFilePath;

    public bool recordingStart = false;
    public bool recordingStop = false;

    void Start()
    {
        // 파일 이름에 날짜와 시간을 추가
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string saveFileName = $"BlendshapeData_{timestamp}.csv";

        // 올바른 저장 경로 설정
        string directoryPath = Path.Combine(Application.dataPath, BlendshapeSavePath);
        saveFilePath = Path.Combine(directoryPath, saveFileName);

        // 폴더가 없으면 생성
        if (!Directory.Exists(directoryPath))
        {
            Directory.CreateDirectory(directoryPath);
        }

        // Blendshape 이름 가져오기
        int blendshapeCount = faceMeshRenderer.sharedMesh.blendShapeCount;

        for (int i = 0; i < blendshapeCount; i++)
        {
            blendshapeNames.Add(faceMeshRenderer.sharedMesh.GetBlendShapeName(i));
        }

        // 파일 초기화
        writer = new StreamWriter(saveFilePath);
        writer.WriteLine("Frame," + string.Join(",", blendshapeNames)); // CSV 헤더 작성
    }

    void FixedUpdate()
    {
        if (recordingStart)
        {
            StartRecording();
        }

        if (recordingStop)
        {
            recordingStart = false;
            StopRecording();
        }


        if (isRecording)
        {
            // Blendshape 값 기록
            List<string> blendshapeValues = new List<string> { Time.frameCount.ToString() };
            for (int i = 0; i < blendshapeNames.Count; i++)
            {
                blendshapeValues.Add(faceMeshRenderer.GetBlendShapeWeight(i).ToString("F2"));
            }
            writer.WriteLine(string.Join(",", blendshapeValues));
        }
    }

    public void StartRecording()
    {
        isRecording = true;
    }

    public void StopRecording()
    {
        isRecording = false;
        if (writer != null)
        {
            writer.Close();
            writer = null;
            Debug.Log($"Blendshape data saved to: {saveFilePath}");
        }
    }

    void OnDestroy()
    {
        if (writer != null)
        {
            StopRecording();
        }
    }
}
