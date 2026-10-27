using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using System.Diagnostics;

public class ARKit2GaussianAvatars : MonoBehaviour
{
    public GaussianSplatting gaussianSplatting;

    public string serverIP = "127.0.0.1";
    public int serverPort = 50051;

    public GameObject headPly;
    public GameObject flameGeneric;
    public GameObject jawBone;
    public GameObject leftEye;
    public GameObject rightEye;

    private TcpClient client;
    private NetworkStream stream;
    private SkinnedMeshRenderer arkitRenderer;
    private SkinnedMeshRenderer flameRenderer;

    public bool start_send = false;

    private StreamedData initStreamedData;

    public int flame_out_dim = 50;

    [SerializeField] private string currentModel = null;

    private Dictionary<string, StreamedData> initCache = new();   // 모델명 -> init 데이터
    private Dictionary<string, int> modelIds = new();             // 모델명 -> modelId

    private GameObject characterAvatar;
    [SerializeField] private GameObject characterAvatarLeftEye;
    [SerializeField] private GameObject characterAvatarRightEye;
    [SerializeField] private GameObject characterAvatarMouth;

    [SerializeField]
    private bool hideCharacterAvatar = true;

    public struct StreamedData
    {
        public float[] expr, jawPose, eyesPose;
        public float[] pos, rot, scale, opacity, sh;
        public int count;
    }
    private class PendingInit
    {
        public string name;
        public StreamedData sd;
        public int count;
    }


    void Start()
    {
        arkitRenderer = headPly.GetComponent<SkinnedMeshRenderer>();
        gaussianSplatting = gaussianSplatting.GetComponent<GaussianSplatting>();

        if (flameGeneric != null)
            flameRenderer = flameGeneric.GetComponent<SkinnedMeshRenderer>();

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


        PreloadModel("ours", "non_bda", "non_mia", "non_mia_non_bda", "mp2flame", "mat");
    }

    void Update()
    {
        // (테스트) 키 입력으로 모델 전환
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha1)) { ActivateModel("mat"); }
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha2)) { ActivateModel("mp2flame"); }
        
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha3)) { ActivateModel("non_bda"); }
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha4)) { ActivateModel("non_mia"); }
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha5)) { ActivateModel("ours"); }

        if (!start_send && Input.GetKeyDown(KeyCode.Alpha6)) { ActivateModel("non_mia_non_bda"); }
        
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha7)) { DeActivateModels(); }
        if (!start_send && Input.GetKeyDown(KeyCode.Alpha8)) { ActivateCharacterModel(); }

        if (!start_send && Input.GetKeyDown(KeyCode.T)) { start_send = true; }
        if (start_send && Input.GetKeyDown(KeyCode.F)) { start_send = false; }

        if (start_send)
        {
            var blendshapes = new Dictionary<string, float>();

            for (int i = 0; i < arkitRenderer.sharedMesh.blendShapeCount; i++)
            {
                string name = arkitRenderer.sharedMesh.GetBlendShapeName(i);
                float weight = arkitRenderer.GetBlendShapeWeight(i);
                blendshapes[name] = weight;
            }

            Stopwatch sw = Stopwatch.StartNew();
            float[] gaussianBytes = SendBlendshapes(blendshapes);
            sw.Stop();
            UnityEngine.Debug.Log($"SendBlendshapes Time: {sw.ElapsedMilliseconds} ms");

            if (gaussianBytes == null) return;

            sw = Stopwatch.StartNew();
            StreamedData streamedData = DataParsing(gaussianBytes);
            sw.Stop();
            UnityEngine.Debug.Log($"DataParsing Time: {sw.ElapsedMilliseconds} ms");

            int id = modelIds[currentModel];

            UnityEngine.Debug.Log($"model ID: {id}, count: {streamedData.count}");

            // SH/opacity는 initCache[currentModel]에서 유지
            streamedData.opacity = initCache[currentModel].opacity;
            streamedData.sh = initCache[currentModel].sh;

            // modelId 타겟 업데이트 (opacity/sh는 그대로 전달 or null 허용 시 생략)
            sw = Stopwatch.StartNew();
            gaussianSplatting.UpdateStreamedModelDataById(id, streamedData.pos, streamedData.rot, streamedData.scale, streamedData.opacity, streamedData.sh, streamedData.count);
            sw.Stop();
            UnityEngine.Debug.Log($"UpdateStreamedModelDataById Time: {sw.ElapsedMilliseconds} ms");

            if (flameRenderer != null)
            {
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
            }
        }
    }

    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
    }

    void PreloadModel(params string[] modelNames)
    {
        var pending = new List<PendingInit>();

        // 1) 모든 모델 init 수신 & 파싱
        foreach (var name in modelNames)
        {
            try
            {
                // 서버에 select 요청해서(또는 list 후 하나씩) 해당 모델의 init 1회 수신
                SendCommandSelect(name, false); // 서버는 init 페이로드를 전송

                float[] init = InitReceiveResponse();
                if (init == null || init.Length == 0)
                {
                    UnityEngine.Debug.LogError($"[{name}] init 수신 실패");
                    continue;
                }

                var sd = InitDataParsing(init); // pos/rot/scale/opacity/sh 분리
                if (sd.count <= 0)
                {
                    UnityEngine.Debug.LogError($"[{name}] 유효하지 않은 gaussian count");
                    continue;
                }

                initCache[name] = sd;
                pending.Add(new PendingInit { name = name, sd = sd, count = sd.count });
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogError($"[{name}] Preload 실패: {e.Message}");
            }
        }

        if (pending.Count == 0)
        {
            UnityEngine.Debug.LogWarning("PreloadModels: 로드 가능한 모델이 없습니다.");
            return;
        }

        // 2) 작은 모델부터 올리도록 정렬
        pending.Sort((a, b) => a.count.CompareTo(b.count));

        // 3) GPU 업로드 (초기 비활성)
        foreach (var item in pending)
        {
            var sd = item.sd;
            int id = gaussianSplatting.AddStreamedModelData(sd.pos, sd.rot, sd.scale, sd.opacity, sd.sh, sd.count);
            if (id <= 0)
            {
                UnityEngine.Debug.LogError($"[{item.name}] AddStreamedModelData 실패");
                continue;
            }

            // 초기엔 비활성(충돌 방지)
            gaussianSplatting.SetStreamedModelActive(id, false);

            modelIds[item.name] = id;
            UnityEngine.Debug.Log($"[{item.name}] 프리로드 완료: count={item.count}, modelId={id}");
        }
    }

    void ActivateModel(string name)
    {
        if (currentModel == name) return;
        // 서버도 같은 모델로 스위칭(추론 일관성). init은 이미 캐시했으니 서버가 다시 보내도 무시 가능.
        SendCommandSelect(name, true);

        // 모든 모델 비활성 → 대상만 활성
        foreach (var kv in modelIds) gaussianSplatting.SetStreamedModelActive(kv.Value, false);
        gaussianSplatting.SetStreamedModelActive(modelIds[name], true);

        currentModel = name;

        if (hideCharacterAvatar)
        {
            if (characterAvatar == null)
            {
                characterAvatar = GameObject.Find("BlendshapeMappingExampleARKIT_A2E - Mirrored_NormalRecalc");
            }
            if (characterAvatar == null)
            {
                characterAvatar = GameObject.Find("BlendshapeMappingExampleARKIT_A2E - Mirrored");
            }
            characterAvatar.SetActive(false);
            characterAvatarLeftEye.SetActive(false);
            characterAvatarRightEye.SetActive(false);
            characterAvatarMouth.SetActive(false);
        }

        UnityEngine.Debug.Log($"[{name}] active 완료");
    }

    void DeActivateModels()
    {
        string name = currentModel;

        // 모든 모델 비활성 → 대상만 활성
        foreach (var kv in modelIds) gaussianSplatting.SetStreamedModelActive(kv.Value, false);
        gaussianSplatting.SetStreamedModelActive(modelIds[name], false);

        currentModel = null;
    }

    void ActivateCharacterModel()
    {
        string name = currentModel;

        // 모든 모델 비활성 → 대상만 활성
        foreach (var kv in modelIds) gaussianSplatting.SetStreamedModelActive(kv.Value, false);
        gaussianSplatting.SetStreamedModelActive(modelIds[name], false);

        currentModel = null;

        if (characterAvatar == null)
        {
            characterAvatar = GameObject.Find("BlendshapeMappingExampleARKIT_A2E - Mirrored_NormalRecalc");
        }
        if (characterAvatar == null)
        {
            characterAvatar = GameObject.Find("BlendshapeMappingExampleARKIT_A2E - Mirrored");
        }
        characterAvatar.SetActive(true);
        characterAvatarLeftEye.SetActive(true);
        characterAvatarRightEye.SetActive(true);
        characterAvatarMouth.SetActive(true);
    }

    // 컨트롤 메시지 전송 (cmd/select)
    private void SendCommandSelect(string modelName, bool no_init)
    {
        var obj = new Dictionary<string, object> {
            {"cmd", "select"},
            {"model", modelName},
            {"no_init", no_init }
        };
        string json = JsonConvert.SerializeObject(obj);
        byte[] message = Encoding.UTF8.GetBytes(json);
        byte[] len = BitConverter.GetBytes(System.Net.IPAddress.HostToNetworkOrder(message.Length));
        stream.Write(len, 0, 4);
        stream.Write(message, 0, message.Length);
    }

    // Call this with a dictionary of 51 ARKit blendshape values (normalized to 0~100)
    private float[] SendBlendshapes(Dictionary<string, float> blendshapes)
    {
        try
        {
            string json = JsonConvert.SerializeObject(blendshapes);
            byte[] message = Encoding.UTF8.GetBytes(json);
            byte[] lengthPrefix = BitConverter.GetBytes(System.Net.IPAddress.HostToNetworkOrder(message.Length));

            stream.Write(lengthPrefix, 0, 4);
            stream.Write(message, 0, message.Length);

            // Receive response
            byte[] lenBuf = new byte[4];
            stream.Read(lenBuf, 0, 4);
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
            UnityEngine.Debug.LogError("Error communicating with server: " + e.Message);
            return null;
        }
    }

    private float[] InitReceiveResponse()
    {
        try
        {
            byte[] lenBuf = new byte[4];
            int readLen = stream.Read(lenBuf, 0, 4);
            if (readLen < 4)
            {
                UnityEngine.Debug.LogError("Failed to read length prefix");
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

    private void InitDataStream(StreamedData gaussianData)
    {
        float[] pos = gaussianData.pos;
        float[] rot = gaussianData.rot;
        float[] scale = gaussianData.scale;
        float[] opacity = gaussianData.opacity;
        float[] sh = gaussianData.sh;
        int count = gaussianData.count;

        UnityEngine.Debug.Log(count);

        gaussianSplatting.InitStreamedModelData(pos, rot, scale, opacity, sh, count);
    }
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


    private StreamedData InitDataParsing(float[] data)
    {
        float[] flameData = new float[109];
        float[] gaussianData = new float[data.Length - 109];

        Buffer.BlockCopy(data, 0, flameData, 0, flameData.Length * sizeof(float));
        Buffer.BlockCopy(data, flameData.Length * sizeof(float), gaussianData, 0, gaussianData.Length * sizeof(float));

        if (gaussianData.Length % 59 != 0)
        {
            UnityEngine.Debug.LogError("Gaussian data length is invalid!");
        }

        float[] expr = new float[100];
        float[] jawPose = new float[3];
        float[] eyesPose = new float[6];

        Buffer.BlockCopy(flameData, 0, expr, 0, 100 * sizeof(float));
        Buffer.BlockCopy(flameData, 100 * sizeof(float), jawPose, 0, 3 * sizeof(float));
        Buffer.BlockCopy(flameData, 103 * sizeof(float), eyesPose, 0, 6 * sizeof(float));

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

    private StreamedData DataParsing(float[] data)
    {
        int flame_data_dim = flame_out_dim + 9;
        float[] flameData = new float[flame_data_dim];
        float[] gaussianData = new float[data.Length - flame_data_dim];

        Buffer.BlockCopy(data, 0, flameData, 0, flameData.Length * sizeof(float));
        Buffer.BlockCopy(data, flameData.Length * sizeof(float), gaussianData, 0, gaussianData.Length * sizeof(float));

        if (gaussianData.Length % 10 != 0)
        {
            UnityEngine.Debug.LogError("Gaussian data length is invalid!");
        }

        float[] expr = new float[flame_out_dim];
        float[] jawPose = new float[3];
        float[] eyesPose = new float[6];

        Buffer.BlockCopy(flameData, 0, expr, 0, flame_out_dim * sizeof(float));
        Buffer.BlockCopy(flameData, flame_out_dim * sizeof(float), jawPose, 0, 3 * sizeof(float));
        Buffer.BlockCopy(flameData, (flame_out_dim + 3) * sizeof(float), eyesPose, 0, 6 * sizeof(float));

        int count = gaussianData.Length / 10;
        float[] pos = new float[count * 3];
        float[] rot = new float[count * 4];
        float[] scale = new float[count * 3];

        for (int i = 0; i < count; i++)
        {
            Array.Copy(gaussianData, i * 10 + 0, pos, i * 3, 3);
            Array.Copy(gaussianData, i * 10 + 3, rot, i * 4, 4);
            Array.Copy(gaussianData, i * 10 + 7, scale, i * 3, 3);
        }

        return new StreamedData
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
    }
}
