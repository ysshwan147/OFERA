using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using System.Diagnostics;

public class ARKit2FLAMEGaussianAvatars : MonoBehaviour
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

    public int flame_out_dim = 100;

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

        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            UnityEngine.Debug.Log("Connected to Python server");

            float[] initialData = InitReceiveResponse();
            if (initialData != null)
            {
                UnityEngine.Debug.Log("Received initial data from server");

                StreamedData streamedData = InitDataParsing(initialData);

                InitDataStream(streamedData);
            }
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
            var blendshapes = new Dictionary<string, float>();

            // headPly blendshapes ���� (0~100 ����)
            for (int i = 0; i < arkitRenderer.sharedMesh.blendShapeCount; i++)
            {
                string name = arkitRenderer.sharedMesh.GetBlendShapeName(i);
                float weight = arkitRenderer.GetBlendShapeWeight(i);
                blendshapes[name] = weight;
            }

            // ������ ���� �� ��� ����
            Stopwatch sw = Stopwatch.StartNew();
            float[] gaussianBytes = SendBlendshapes(blendshapes);
            if (gaussianBytes == null) return;

            sw.Stop();
            UnityEngine.Debug.Log($"SendBlendshapes Time: {sw.ElapsedMilliseconds} ms");

            sw = Stopwatch.StartNew();
            StreamedData streamedData = DataParsing(gaussianBytes);
            sw.Stop();
            UnityEngine.Debug.Log($"DataParsing Time: {sw.ElapsedMilliseconds} ms");

            sw = Stopwatch.StartNew();
            UpdateDataStream(streamedData);
            sw.Stop();
            UnityEngine.Debug.Log($"UpdateDataStream Time: {sw.ElapsedMilliseconds} ms");

            // Apply FLAME expression blendshapes (indices 0~99)
            for (int i = 0; i < Mathf.Min(flame_out_dim, flameRenderer.sharedMesh.blendShapeCount); i++)
            {
                flameRenderer.SetBlendShapeWeight(300 + i, streamedData.expr[i] * 100f);
            }

            // Apply jaw pose (Euler angles)
            jawBone.transform.localRotation = Quaternion.Euler(
                streamedData.jawPose[0],
                - streamedData.jawPose[1],
                - streamedData.jawPose[2]
            );

            // Apply eye poses
            leftEye.transform.localRotation = Quaternion.Euler(
                streamedData.eyesPose[0],
                - streamedData.eyesPose[1],
                - streamedData.eyesPose[2]
            );
            rightEye.transform.localRotation = Quaternion.Euler(
                streamedData.eyesPose[3],
                - streamedData.eyesPose[4],
                - streamedData.eyesPose[5]
            );
        }

    }

    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
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

            // byte[] �� float[] ��ȯ
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

            // byte[] �� float[] ��ȯ
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
