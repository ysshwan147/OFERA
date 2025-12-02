using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Net.Sockets;

public class BsMpRecorder : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int serverPort = 50051;

    private TcpClient client;
    private NetworkStream stream;

    public bool start_send = false;

    public Camera cam;

    public SkinnedMeshRenderer faceMeshRenderer; // Blendshape 데이터를 가지고 있는 렌더러
    private StreamWriter writer;
    private List<string> blendshapeNames = new();

    // 저장 경로 수정 (Assets 폴더 내부 RecordedBlendshapes 사용)
    public string BlendshapeSavePath = "RecordedBlendshapes";
    private string saveFilePath;

    private int blendshapeCount;

    // Unity에서 사용 중인 블렌드셰입 이름 순서대로 리스트 생성
    private readonly List<string> unityBlendshapeOrder = new List<string> {
    "eyeWideLeft", "eyeWideRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "cheekSquintLeft", "cheekSquintRight",
    "mouthRollUpper", "eyeBlinkLeft", "eyeBlinkRight", "mouthClose", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthPressLeft", "mouthPressRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthDimpleLeft", "mouthDimpleRight", "jawLeft", "jawRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthLeft", "mouthRight", "mouthPucker", "jawForward", "jawOpen",
    "mouthRollLower", "cheekPuff", "noseSneerRight", "noseSneerLeft", "mouthShrugLower", "mouthShrugUpper",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"
};

    // 서버에서 요구하는 블렌드셰입 순서
    private readonly List<string> serverBlendshapeOrder = new List<string> {
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
};

    private readonly List<string> serverBlendshapeOrder_flipped = new List<string> {
    "browDownRight", "browDownLeft", "browInnerUp", "browOuterUpRight", "browOuterUpLeft", "cheekPuff",
    "cheekSquintRight", "cheekSquintLeft", "eyeBlinkRight", "eyeBlinkLeft", "eyeLookDownRight", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookInLeft", "eyeLookOutRight", "eyeLookOutLeft", "eyeLookUpRight", "eyeLookUpLeft",
    "eyeSquintRight", "eyeSquintLeft", "eyeWideRight", "eyeWideLeft", "jawForward", "jawRight", "jawOpen", "jawLeft",
    "mouthClose", "mouthDimpleRight", "mouthDimpleLeft", "mouthFrownRight", "mouthFrownLeft", "mouthFunnel",
    "mouthRight", "mouthLowerDownRight", "mouthLowerDownLeft", "mouthPressRight", "mouthPressLeft",
    "mouthPucker", "mouthLeft", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileRight", "mouthSmileLeft", "mouthStretchRight", "mouthStretchLeft", "mouthUpperUpRight",
    "mouthUpperUpLeft", "noseSneerRight", "noseSneerLeft"
};


    // 이름 → 인덱스 매핑
    private Dictionary<string, int> blendshapeNameToIndex;
    private int[] serverOrderedIndices;

    private void InitializeBlendshapeIndexMap()
    {
        blendshapeNameToIndex = new Dictionary<string, int>();
        for (int i = 0; i < unityBlendshapeOrder.Count; i++)
        {
            blendshapeNameToIndex[unityBlendshapeOrder[i]] = i;
        }

        serverOrderedIndices = new int[serverBlendshapeOrder_flipped.Count];
        for (int i = 0; i < serverBlendshapeOrder_flipped.Count; i++)
        {
            string name = serverBlendshapeOrder_flipped[i];
            if (blendshapeNameToIndex.TryGetValue(name, out int index))
            {
                serverOrderedIndices[i] = index;
            }
            else
            {
                serverOrderedIndices[i] = -1;  // 미정의된 경우
            }
        }
    }

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
        blendshapeCount = faceMeshRenderer.sharedMesh.blendShapeCount;

        for (int i = 0; i < blendshapeCount; i++)
        {
            blendshapeNames.Add(faceMeshRenderer.sharedMesh.GetBlendShapeName(i));
        }

        // 리스트를 하나의 문자열로 합치기 (쉼표로 구분)
        string joinedString = string.Join(", ", blendshapeNames);

        // Debug.Log로 출력
        Debug.Log("Blendshape Values: " + joinedString);

        InitializeBlendshapeIndexMap();

        // 파일 초기화
        // writer = new StreamWriter(saveFilePath);
        // writer.WriteLine("Frame," + string.Join(",", blendshapeNames)); // CSV 헤더 작성

        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            UnityEngine.Debug.Log("Connected to Python server");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Connection failed: " + e.Message);
        }

        
    }

    void Update()
    {
        if (start_send)
        {
            SendFrameData();
        }

    }

    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
        if (writer != null)
        {
            writer.Close();
            writer = null;
            Debug.Log($"Blendshape data saved to: {saveFilePath}");
        }
    }

    private void SendFrameData()
    {
        try
        {
            // 1. 카메라 렌더링 → PNG 바이트 배열 생성
            RenderTexture rt = new RenderTexture(512, 512, 24);
            cam.targetTexture = rt;
            Texture2D tex = new Texture2D(512, 512, TextureFormat.RGB24, false);
            cam.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, 512, 512), 0, 0);
            tex.Apply();
            cam.targetTexture = null;
            RenderTexture.active = null;
            UnityEngine.Object.Destroy(rt);
            byte[] pngBytes = tex.EncodeToPNG();
            UnityEngine.Object.Destroy(tex);

            // 2. Blendshape 값들 (float32 → byte[])
            List<float> blendshapeList = new List<float>();
            foreach (int index in serverOrderedIndices)
            {
                blendshapeList.Add(index >= 0 ? faceMeshRenderer.GetBlendShapeWeight(index) : 0f);
            }
            byte[] blendshapeBytes = new byte[blendshapeList.Count * 4];
            Buffer.BlockCopy(blendshapeList.ToArray(), 0, blendshapeBytes, 0, blendshapeBytes.Length);

            // 3. 전체 메시지 구조: [blendshape_len(4)][image_len(4)][blendshape][image]
            byte[] blendshapeLen = BitConverter.GetBytes(System.Net.IPAddress.HostToNetworkOrder(blendshapeBytes.Length));
            byte[] imageLen = BitConverter.GetBytes(System.Net.IPAddress.HostToNetworkOrder(pngBytes.Length));

            // 4. 전송
            stream.Write(blendshapeLen, 0, 4);
            stream.Write(imageLen, 0, 4);
            stream.Write(blendshapeBytes, 0, blendshapeBytes.Length);
            stream.Write(pngBytes, 0, pngBytes.Length);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Error sending frame data: " + e.Message);
        }
    }

}
