using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;

[System.Serializable]
public class ASLMessage
{
    public string first;
    public float first_conf;
    public string second;
    public float second_conf;
}

public class pyReceiver : MonoBehaviour
{
    [Header("Current ASL Predictions")]
    public string firstLetter = "";
    public float firstConfidence = 0f;
    public string secondLetter = "";
    public float secondConfidence = 0f;

    private PullSocket socket;
    private Thread zmqThread;
    private bool running = true;

    void Start()
    {
        StartZMQ();
    }

    void StartZMQ()
    {
        zmqThread = new Thread(() =>
        {
            socket = new PullSocket();
            socket.Connect("tcp://localhost:5555");

            while (running)
            {
                if (socket.TryReceiveFrameString(System.TimeSpan.FromMilliseconds(100), out string message))
                {
                    ASLMessage asl = JsonUtility.FromJson<ASLMessage>(message);

                    firstLetter = asl.first;
                    firstConfidence = asl.first_conf;
                    secondLetter = asl.second;
                    secondConfidence = asl.second_conf;
                }
            }
            socket?.Close();
        })
        { IsBackground = true };

        zmqThread.Start();
    }

    void OnApplicationQuit()
    {
        running = false;
        zmqThread?.Join(1000);
        socket?.Close();
        NetMQConfig.Cleanup();
    }
}
