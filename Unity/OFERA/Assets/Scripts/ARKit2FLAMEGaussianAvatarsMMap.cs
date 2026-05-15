using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using UnityEngine;

public class ARKit2FLAMEGaussianAvatarsMMap : MonoBehaviour
{
    public GaussianSplatting gaussianSplatting;

    [Header("Python TCP Server (for ARKit JSON only)")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 50051;

    [Header("Shared Memory (MemoryMappedFile)")]
    // Python 쪽과 반드시 동일한 이름을 사용해야 합니다.
    public string sharedMemoryName = @"Local\GaussianSharedBuffer";

    // readyFlag(4) + count(4) 이후에 float 데이터가 이어집니다.
    // float32 기준: expr(100) + jaw(3) + eyes(6) + gaussians(count * 10)
    // 여기서 flame_out_dim은 expr 차원과 같아야 합니다.
    public int flame_out_dim = 50;

    [Header("Avatar Parts")]
    public GameObject headPly;
    public GameObject flameGeneric;
    public GameObject jawBone;
    public GameObject leftEye;
    public GameObject rightEye;

    [Header("Control")]
    public bool start_send = false;

    private TcpClient client;
    private NetworkStream stream;
    private SkinnedMeshRenderer arkitRenderer;
    private SkinnedMeshRenderer flameRenderer;

    // Shared memory
    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor accessor;

    private StreamedData initStreamedData;

    public struct StreamedData
    {
        public float[] expr, jawPose, eyesPose;
        public float[] pos, rot, scale, opacity, sh;
        public int count;
    }

    void Start()
    {
        arkitRenderer = headPly.GetComponent<SkinnedMeshRenderer>();
        flameRenderer = flameGeneric.GetComponent<SkinnedMeshRenderer>();
        gaussianSplatting = gaussianSplatting.GetComponent<GaussianSplatting>();

        // 1) Python TCP 서버 연결 (블렌드셰이프 전송용)
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            UnityEngine.Debug.Log("Connected to Python server (TCP for ARKit JSON)");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("TCP connection failed: " + e.Message);
        }

        // 2) 초기 init payload (Gaussian + opacity/SH) 는 기존처럼 TCP로 한 번만 받습니다.
        //    이 데이터로 GaussianSplatting 쪽에 모델을 올려놓고, opacity/SH 캐시를 만들어둡니다.
        if (stream != null)
        {
            float[] initialData = InitReceiveResponse();
            if (initialData != null)
            {
                UnityEngine.Debug.Log("Received initial data from server over TCP");
                StreamedData streamedData = InitDataParsing(initialData); // expr/jaw/eyes + pos/rot/scale/opacity/SH 분해 :contentReference[oaicite:1]{index=1}
                InitDataStream(streamedData);                             // GaussianSplatting.InitStreamedModelData 호출 :contentReference[oaicite:2]{index=2}
            }
        }

        // 3) Python이 생성할 MemoryMappedFile 열기 (공유 메모리)
        TryOpenSharedMemory();
    }

    void Update()
    {
        if (!start_send) return;

        if (stream == null || !stream.CanWrite)
        {
            UnityEngine.Debug.LogWarning("TCP stream is not ready.");
            return;
        }

        if (mmf == null || accessor == null)
        {
            // 아직 Python이 MMap을 만들지 않았을 수 있으니, 계속 재시도
            TryOpenSharedMemory();
            if (mmf == null)
            {
                UnityEngine.Debug.LogWarning("Shared memory not available yet.");
                return;
            }
        }

        // 1) 현재 ARKit 블렌드셰이프를 읽어서 Python으로 JSON 전송
        var blendshapes = new Dictionary<string, float>();
        for (int i = 0; i < arkitRenderer.sharedMesh.blendShapeCount; i++)
        {
            string name = arkitRenderer.sharedMesh.GetBlendShapeName(i);
            float weight = arkitRenderer.GetBlendShapeWeight(i);
            blendshapes[name] = weight;
        }

        Stopwatch sw = Stopwatch.StartNew();
        SendBlendshapesJsonOnly(blendshapes);
        sw.Stop();
        UnityEngine.Debug.Log($"SendBlendshapesJsonOnly Time: {sw.ElapsedMilliseconds} ms");

        // 2) Python이 shared memory에 새 프레임을 썼는지 ready 플래그 확인
        sw = Stopwatch.StartNew();
        StreamedData streamedData;
        bool ok = TryReadFromSharedMemory(out streamedData);
        sw.Stop();
        if (!ok)
        {
            // 아직 새 프레임이 준비 안 된 경우도 있으니 조용히 패스
            // UnityEngine.Debug.Log("No new shared-memory frame ready.");
            return;
        }
        UnityEngine.Debug.Log($"ReadFromSharedMemory Time: {sw.ElapsedMilliseconds} ms");

        // 3) GaussianSplatting 플러그인으로 데이터 전달
        sw = Stopwatch.StartNew();
        UpdateDataStream(streamedData);
        sw.Stop();
        UnityEngine.Debug.Log($"UpdateDataStream Time: {sw.ElapsedMilliseconds} ms");

        // 4) FLAME mesh / bones에 반영 (기존 코드 재사용) :contentReference[oaicite:3]{index=3}
        // Apply FLAME expression blendshapes (indices 0~flame_out_dim-1)
        for (int i = 0; i < Mathf.Min(flame_out_dim, flameRenderer.sharedMesh.blendShapeCount); i++)
        {
            flameRenderer.SetBlendShapeWeight(300 + i, streamedData.expr[i] * 100f);
        }

        // Apply jaw pose (Euler angles)
        jawBone.transform.localRotation = Quaternion.Euler(
            streamedData.jawPose[0],
            -streamedData.jawPose[1],
            -streamedData.jawPose[2]
        );

        // Apply eye poses
        leftEye.transform.localRotation = Quaternion.Euler(
            streamedData.eyesPose[0],
            -streamedData.eyesPose[1],
            -streamedData.eyesPose[2]
        );
        rightEye.transform.localRotation = Quaternion.Euler(
            streamedData.eyesPose[3],
            -streamedData.eyesPose[4],
            -streamedData.eyesPose[5]
        );

        UnityEngine.Debug.Log($"End update");
        UnityEngine.Debug.Log($"End update");
    }

    void OnApplicationQuit()
    {
        accessor?.Dispose();
        mmf?.Dispose();
        stream?.Close();
        client?.Close();
    }

    #region TCP 통신 (초기 init + ARKit JSON)

    // 초기 init payload만 받는 함수 (기존 코드 그대로 재사용)
    private float[] InitReceiveResponse()
    {
        try
        {
            byte[] lenBuf = new byte[4];
            int readLen = stream.Read(lenBuf, 0, 4);
            if (readLen < 4)
            {
                UnityEngine.Debug.LogError("Failed to read length prefix for initial message");
                return null;
            }

            int msgLen = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(lenBuf, 0));
            byte[] msgBuf = new byte[msgLen];
            int read = 0;
            while (read < msgLen)
            {
                read += stream.Read(msgBuf, read, msgLen - read);
            }

            int floatCount = msgLen / 4;
            float[] result = new float[floatCount];
            Buffer.BlockCopy(msgBuf, 0, result, 0, msgLen);

            return result;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Error receiving initial message: " + e.Message);
            return null;
        }
    }

    // ARKit 블렌드셰이프 JSON만 보내고, 응답은 기다리지 않는 버전
    private void SendBlendshapesJsonOnly(Dictionary<string, float> blendshapes)
    {
        try
        {
            string json = JsonConvert.SerializeObject(blendshapes);
            byte[] message = Encoding.UTF8.GetBytes(json);
            byte[] lengthPrefix = BitConverter.GetBytes(System.Net.IPAddress.HostToNetworkOrder(message.Length));

            // 기존 구조 그대로: [4바이트 길이] + [JSON 바디]
            stream.Write(lengthPrefix, 0, 4);
            stream.Write(message, 0, message.Length);
            stream.Flush();
            // 가우시안 float[] 응답은 TCP로 더 이상 받지 않고,
            // Python이 shared memory에 쓰는 것을 Unity가 직접 읽습니다.
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Error sending blendshapes JSON: " + e.Message);
        }
    }

    #endregion

    #region GaussianSplatting 연동 (초기/업데이트)

    // 초기 Gaussian 데이터(pos/rot/scale/opacity/SH)를 플러그인으로 넘기기 :contentReference[oaicite:4]{index=4}
    private void InitDataStream(StreamedData gaussianData)
    {
        float[] pos = gaussianData.pos;
        float[] rot = gaussianData.rot;
        float[] scale = gaussianData.scale;
        float[] opacity = gaussianData.opacity;
        float[] sh = gaussianData.sh;
        int count = gaussianData.count;

        UnityEngine.Debug.Log($"[InitDataStream] Gaussian count = {count}");

        gaussianSplatting.InitStreamedModelData(pos, rot, scale, opacity, sh, count);
    }

    // 매 프레임 Gaussian 데이터 업데이트
    private void UpdateDataStream(StreamedData gaussianData)
    {
        float[] pos = gaussianData.pos;
        float[] rot = gaussianData.rot;
        float[] scale = gaussianData.scale;
        float[] opacity = gaussianData.opacity;
        float[] sh = gaussianData.sh;
        int count = gaussianData.count;

        gaussianSplatting.UpdateStreamedModelData(pos, rot, scale, opacity, sh, count);
    }

    #endregion

    #region 초기 init 데이터 파싱 (expr/jaw/eyes + full gaussian) :contentReference[oaicite:5]{index=5}

    private StreamedData InitDataParsing(float[] data)
    {
        // 초기 init은 expr(100) + jaw(3) + eyes(6) = 109 를 FLAME로 사용
        float[] flameData = new float[59];
        float[] gaussianData = new float[data.Length - 59];

        Buffer.BlockCopy(data, 0, flameData, 0, flameData.Length * sizeof(float));
        Buffer.BlockCopy(data, flameData.Length * sizeof(float), gaussianData, 0, gaussianData.Length * sizeof(float));

        if (gaussianData.Length % 59 != 0)
        {
            UnityEngine.Debug.LogError("Init Gaussian data length is invalid!");
        }

        float[] expr = new float[50];
        float[] jawPose = new float[3];
        float[] eyesPose = new float[6];

        Buffer.BlockCopy(flameData, 0, expr, 0, 50 * sizeof(float));
        Buffer.BlockCopy(flameData, 50 * sizeof(float), jawPose, 0, 3 * sizeof(float));
        Buffer.BlockCopy(flameData, 53 * sizeof(float), eyesPose, 0, 6 * sizeof(float));

        int count = gaussianData.Length / 59;
        float[] pos = new float[count * 3];
        float[] rot = new float[count * 4];
        float[] scale = new float[count * 3];
        float[] opacity = new float[count];
        float[] sh = new float[count * 48];

        for (int i = 0; i < count; i++)
        {
            Array.Copy(gaussianData, i * 59 + 0, pos, i * 3, 3);
            Array.Copy(gaussianData, i * 59 + 3, rot, i * 4, 4);
            Array.Copy(gaussianData, i * 59 + 7, scale, i * 3, 3);
            opacity[i] = gaussianData[i * 59 + 10];
            Array.Copy(gaussianData, i * 59 + 11, sh, i * 48, 48);
        }

        initStreamedData = new StreamedData
        {
            expr = expr,
            jawPose = jawPose,
            eyesPose = eyesPose,
            pos = pos,
            rot = rot,
            scale = scale,
            opacity = opacity,
            sh = sh,
            count = count
        };

        return initStreamedData;
    }

    #endregion

    #region SharedMemory(MMap) 열기 & 읽기

    private void TryOpenSharedMemory()
    {
        if (mmf != null && accessor != null) return;

        try
        {
            mmf = MemoryMappedFile.OpenExisting(sharedMemoryName);
            accessor = mmf.CreateViewAccessor();
            UnityEngine.Debug.Log($"Opened shared memory: {sharedMemoryName}");
        }
        catch (Exception e)
        {
            // 아직 Python이 MMap을 생성하지 않았을 수도 있으니, 에러를 너무 자주 찍진 않도록 합니다.
            UnityEngine.Debug.LogWarning($"Failed to open shared memory '{sharedMemoryName}': {e.Message}");
            mmf = null;
            accessor = null;
        }
    }

    /// <summary>
    /// Python이 shared memory에 쓴 1프레임을 읽어서 StreamedData로 변환합니다.
    /// layout:
    ///   int readyFlag (0: Unity 읽는중/미완료, 1: 새 데이터 준비 완료)
    ///   int count
    ///   float expr[flame_out_dim]
    ///   float jaw[3]
    ///   float eyes[6]
    ///   float pos[count * 3]
    ///   float rot[count * 4]
    ///   float scale[count * 3]
    /// </summary>
    private bool TryReadFromSharedMemory(out StreamedData streamed)
    {
        streamed = default;

        if (accessor == null || initStreamedData.count <= 0)
        {
            return false;
        }

        // 0: readyFlag
        int ready = accessor.ReadInt32(0);
        if (ready == 0)
        {
            // 새 프레임이 아직 준비되지 않음
            return false;
        }

        // 4: count (Python과 Unity 양쪽이 같은 count를 사용한다고 가정)
        int count = accessor.ReadInt32(4);
        if (count != initStreamedData.count)
        {
            UnityEngine.Debug.LogWarning($"Shared memory count({count}) != initStreamedData.count({initStreamedData.count})");
        }

        // float 데이터 읽기
        long offset = 8;

        float[] expr = new float[flame_out_dim];
        accessor.ReadArray(offset, expr, 0, flame_out_dim);
        offset += flame_out_dim * sizeof(float);

        float[] jawPose = new float[3];
        accessor.ReadArray(offset, jawPose, 0, 3);
        offset += 3 * sizeof(float);

        float[] eyesPose = new float[6];
        accessor.ReadArray(offset, eyesPose, 0, 6);
        offset += 6 * sizeof(float);

        int posCount = count * 3;
        int rotCount = count * 4;
        int scaleCount = count * 3;

        float[] pos = new float[posCount];
        float[] rot = new float[rotCount];
        float[] scale = new float[scaleCount];

        accessor.ReadArray(offset, pos, 0, posCount);
        offset += posCount * sizeof(float);

        accessor.ReadArray(offset, rot, 0, rotCount);
        offset += rotCount * sizeof(float);

        accessor.ReadArray(offset, scale, 0, scaleCount);
        offset += scaleCount * sizeof(float);

        // opacity / sh 는 init 때 받은 것을 재사용 (Python에서도 변하지 않는다고 가정)
        streamed = new StreamedData
        {
            expr = expr,
            jawPose = jawPose,
            eyesPose = eyesPose,
            pos = pos,
            rot = rot,
            scale = scale,
            opacity = initStreamedData.opacity,
            sh = initStreamedData.sh,
            count = count
        };

        // 이 프레임 읽기 완료를 Python에게 알려주기 위해 readyFlag = 0 으로 되돌립니다.
        accessor.Write(0, 0);

        return true;
    }

    #endregion
}
