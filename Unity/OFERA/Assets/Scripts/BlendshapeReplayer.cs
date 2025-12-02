using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class BlendshapeReplayer : MonoBehaviour
{
    public SkinnedMeshRenderer faceMeshRenderer;
    public string BlendshapeSavePath = "RecordedBlendshapes";
    public string loadFilePath = "BlendshapeData.csv";
    private List<Dictionary<string, float>> frameData;
    private List<string> blendshapeNames;
    private int currentFrame = 0;

    public bool startReplay = false;
    public bool reset = false;
    public bool repeat = false;
    public bool avaturn = false;

    public float factor = 100.0f;

    [Range(0, 1000)] public int manualFrame = 0; // 슬라이더로 제어
    public bool manualControl = false; // 슬라이더 수동 조작 모드 활성화 여부

    private int lastManualFrame = -1;

    void Start()
    {
        // if (avaturn) factor = 0.01f;

        string directoryPath = Path.Combine(Application.dataPath, BlendshapeSavePath);
        string saveFilePath = Path.Combine(directoryPath, loadFilePath);

        Dictionary<string, string> blendShapeMap = new()
        {
            { "bls_head.browDownLeft", "browDownLeft" },
            { "bls_head.browDownRight", "browDownRight" },
            { "bls_head.cheekSquintLeft", "cheekSquintLeft" },
            { "bls_head.cheekSquintRight", "cheekSquintRight" },
            { "bls_head.mouthShrugLower", "mouthShrugLower" },
            { "bls_head.mouthShrugUpper", "mouthShrugUpper" },
            { "bls_head.mouthDimpleLeft", "mouthDimpleLeft" },
            { "bls_head.mouthDimpleRight", "mouthDimpleRight" },
            { "bls_head.eyeBlinkLeft", "eyeBlinkLeft" },
            { "bls_head.eyeBlinkRight", "eyeBlinkRight" },
            { "bls_head.eyeLookDownLeft", "eyeLookDownLeft" },
            { "bls_head.eyeLookDownRight", "eyeLookDownRight" },
            { "bls_head.eyeLookOutLeft", "eyeLookOutLeft" },
            { "bls_head.eyeLookInRight", "eyeLookInRight" },
            { "bls_head.eyeLookInLeft", "eyeLookInLeft" },
            { "bls_head.eyeLookOutRight", "eyeLookOutRight" },
            { "bls_head.eyeLookUpLeft", "eyeLookUpLeft" },
            { "bls_head.eyeLookUpRight", "eyeLookUpRight" },
            { "bls_head.jawOpen", "jawOpen" },
            { "bls_head.jawLeft", "jawLeft" },
            { "bls_head.jawRight", "jawRight" },
            { "bls_head.jawForward", "jawForward" },
            { "bls_head.eyeSquintLeft", "eyeSquintLeft" },
            { "bls_head.eyeSquintRight", "eyeSquintRight" },
            { "bls_head.mouthFrownLeft", "mouthFrownLeft" },
            { "bls_head.mouthFrownRight", "mouthFrownRight" },
            { "bls_head.mouthSmileRight", "mouthSmileRight" },
            { "bls_head.mouthSmileLeft", "mouthSmileLeft" },
            { "bls_head.mouthPressLeft", "mouthPressLeft" },
            { "bls_head.mouthPressRight", "mouthPressRight" },
            { "bls_head.mouthStretchLeft", "mouthStretchLeft" },
            { "bls_head.mouthStretchRight", "mouthStretchRight" },
            { "bls_head.mouthLowerDownLeft", "mouthLowerDownLeft" },
            { "bls_head.mouthLowerDownRight", "mouthLowerDownRight" },
            { "bls_head.mouthLeft", "mouthLeft" },
            { "bls_head.mouthRight", "mouthRight" },
            { "bls_head.noseSneerLeft", "noseSneerLeft" },
            { "bls_head.noseSneerRight", "noseSneerRight" },
            { "bls_head.browOuterUpLeft", "browOuterUpLeft" },
            { "bls_head.browOuterUpRight", "browOuterUpRight" },
            { "bls_head.eyeWideLeft", "eyeWideLeft" },
            { "bls_head.eyeWideRight", "eyeWideRight" },
            { "bls_head.mouthUpperUpLeft", "mouthUpperUpLeft" },
            { "bls_head.mouthUpperUpRight", "mouthUpperUpRight" },
            { "bls_head.mouthClose", "mouthClose" },
            { "bls_head.mouthFunnel", "mouthFunnel" },
            { "bls_head.mouthPucker", "mouthPucker" },
            { "bls_head.mouthRollLower", "mouthRollLower" },
            { "bls_head.mouthRollUpper", "mouthRollUpper" },
            { "bls_head.browInnerUp", "browInnerUp" },
            { "bls_head.cheekPuff", "cheekPuff" }
        };

        frameData = new List<Dictionary<string, float>>();
        using (StreamReader reader = new(saveFilePath))
        {
            string headerLine = reader.ReadLine();
            blendshapeNames = new List<string>(headerLine.Split(','));
            blendshapeNames.RemoveAt(0);

            if (avaturn)
            {
                for (int i = 0; i < blendshapeNames.Count; i++)
                {
                    blendshapeNames[i] = blendShapeMap[blendshapeNames[i]];
                }
            }

            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] values = line.Split(',');
                Dictionary<string, float> frameValues = new();

                for (int i = 1; i < values.Length; i++)
                {
                    frameValues[blendshapeNames[i - 1]] = float.Parse(values[i]);
                }
                frameData.Add(frameValues);
            }
        }

        // 슬라이더 범위 재설정
        manualFrame = 0;
        lastManualFrame = -1;
    }

    void FixedUpdate()
    {
        if (startReplay && !manualControl)
        {
            if (currentFrame < frameData.Count)
            {
                ApplyBlendshapeValues(frameData[currentFrame]);
                currentFrame++;
            }
            else if (repeat)
            {
                currentFrame = 0;
            }
        }

        if (reset)
        {
            startReplay = false;
            currentFrame = 0;
            InitializeBlendshapeValues();
            reset = false;
        }
    }

    void Update()
    {
        if (manualControl && manualFrame != lastManualFrame && manualFrame < frameData.Count)
        {
            ApplyBlendshapeValues(frameData[manualFrame]);
            lastManualFrame = manualFrame;
        }
    }

    private void ApplyBlendshapeValues(Dictionary<string, float> blendshapeValues)
    {
        foreach (var kvp in blendshapeValues)
        {
            int index = faceMeshRenderer.sharedMesh.GetBlendShapeIndex(kvp.Key);
            if (index >= 0)
            {
                faceMeshRenderer.SetBlendShapeWeight(index, kvp.Value * factor);
            }
        }
    }

    private void InitializeBlendshapeValues()
    {
        for (int i = 0; i < blendshapeNames.Count; i++)
        {
            int index = faceMeshRenderer.sharedMesh.GetBlendShapeIndex(blendshapeNames[i]);
            if (index >= 0)
            {
                faceMeshRenderer.SetBlendShapeWeight(index, 0);
            }
        }
    }
}
