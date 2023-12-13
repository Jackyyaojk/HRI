using System;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;

public class Commander : MonoBehaviour
{
    public GameObject Sphere;
    public GameObject initManager;
    GameObject[] gameObjects;
    // private string url = "http://192.168.1.12:8010/robotInit.txt";
    // private string url_update = "http://192.168.1.12:8010/robotUpdate.txt";
    private string url = "http://192.168.0.184:8010/robotInit.txt";
    private string url_update = "http://192.168.0.184:8010/robotUpdate.txt";

    private int initialized = 0;
    private Boolean initialUrlGot = false;
    private Vector3[] locations;
    private Matrix4x4 transMatrix;
    private Matrix4x4 invTransMatrix;
    private Matrix4x4 H;
    private string placedCoordinate;
    private float latch_point;
    private Boolean trajectories = false;
    private int[] traj_nums = { 0, 0 };
    private int failCount = 0;
    private Gradient gradient;
    private GradientColorKey[] colorKey;
    private GradientAlphaKey[] alphaKey;
    private Boolean task1;

    // Start is called before the first frame update
    void Start()
    {
        Initialize();
    }

    // Update is called once per frame
    void Update()
    {
        if (initialized == 0)
        {
            Initialize();
        }
        else if (initialized == 1)
        {
            if (!initialUrlGot)
            {
                StartCoroutine(GetRequestInitial());
                initialUrlGot = true;
            }

            if (initialUrlGot)
            {
                return;
            }

        }

        else if (initialized == 2)
        {
            StartCoroutine(GetRequestUpdate());
        }

    }

    public void Initialize()
    {
        //Debug.Log(initManager.GetComponent<InitScript>().objectCounter);
        if (initManager.GetComponent<InitScript>().objectCounter == 4)
        {
            locations = initManager.GetComponent<InitScript>().objectLocations;
            transMatrix = initManager.GetComponent<InitScript>().calculateTransMatrix(locations);
            invTransMatrix = transMatrix.inverse;
            initialized = 1;
            initManager.GetComponent<InitScript>().destroySpheres();
            Debug.Log("Initialized() Complete");
            Debug.Log("Inverse Transformation Matrix:");
            Debug.Log(invTransMatrix.ToString());
            Debug.Log("Regular Transformation Matrix:");
            Debug.Log(transMatrix);

        }
    }

    IEnumerator GetRequestInitial()
    {
        //Submit Web Request and log incoming errors and data
        //Debug.Log("Running Request Intitialization");
        //initManager.GetComponent<InitScript>().guiText.SetText("");

        UnityWebRequest r = UnityWebRequest.Get(url);
        yield return r.SendWebRequest();
        if (r.isNetworkError || r.isHttpError) // If you are not getting the server requests, continue until you do
        {
            initialUrlGot = false;
            System.Threading.Thread.Sleep(100);
            yield break;
        }
        else
        {
            if (r.downloadHandler.text.Equals("Done"))
            {
                Debug.Log("Got Done");
                initialUrlGot = false;
                System.Threading.Thread.Sleep(100);
                yield break;

            }       
            if (gameObjects != null)
            {
                deinitialize();
                task1 = false;
                initialUrlGot = false;
                System.Threading.Thread.Sleep(100);
                yield break;
            }
            Debug.Log(r.downloadHandler.text);
            //Parse data and create new objects
            string[] lines = r.downloadHandler.text.Split('\n');
            r.Dispose();
            gameObjects = new GameObject[lines.Length];
            Debug.Log(lines.Length);
            for (int i = 0; i < lines.Length; i++)
            {
                string[] words = lines[i].Split('\t');
                if (words.Length > 6)
                {
                    task1 = true;
                }

                    //Instantiate current location object           
                if (i == 0)
                {
                    gradient = new Gradient();
                    colorKey = new GradientColorKey[3];
                    colorKey[0].color = Color.black;
                    colorKey[0].time = 0.1f;
                    colorKey[1].color = Color.grey;
                    colorKey[1].time = float.Parse(words[0]) / 2.5f;
                    colorKey[2].color = new Color(1.0f, 0.5f, 0.0f, 1.0f);
                    colorKey[2].time = float.Parse(words[0]);
                    alphaKey = new GradientAlphaKey[3];
                    alphaKey[0].alpha = 0.0f;
                    alphaKey[0].time = 0.1f;
                    alphaKey[1].alpha = 1.0f;
                    alphaKey[1].time = 0.2f;
                    alphaKey[2].alpha = 1.0f;
                    alphaKey[2].time = 1.0f;
                    gradient.SetKeys(colorKey, alphaKey);
                                 
                    latch_point = float.Parse(words[0]);
                    gameObjects[i] = Instantiate(Sphere, coordTransform(new Vector3(float.Parse(words[1]), float.Parse(words[2]), float.Parse(words[3]))), Quaternion.identity);
                    if (words.Length > 6)
                    {
                        gameObjects[i].transform.rotation = getArmRotation(new Vector3(float.Parse(words[4]), float.Parse(words[5]), float.Parse(words[6])));
                    }

                }
                else
                {
                    // If trajectory, make child of parent goal. If not a trajectory it has to be a parent goal.
                    if (words[0] == "traj")
                    {
                        gameObjects[i] = Instantiate(Sphere, coordTransform(new Vector3(float.Parse(words[2]), float.Parse(words[3]), float.Parse(words[4]))), Quaternion.identity);
                        GameObject curr = gameObjects[i];
                        GameObject o = gameObjects[int.Parse(words[1])];
                        curr.name = String.Format("traj {0}", i);
                        curr.transform.parent = o.transform;
                        trajectories = true;
                    }
                    else // else, its a goal
                    {
                        gameObjects[i] = Instantiate(Sphere, coordTransform(new Vector3(float.Parse(words[0]), float.Parse(words[1]), float.Parse(words[2]))), Quaternion.identity);
                        gameObjects[i].name = String.Format("goal {0}", i);
                        if (words.Length > 6)
                        {
                            gameObjects[i].transform.rotation = getArmRotation(new Vector3(float.Parse(words[3]), float.Parse(words[4]), float.Parse(words[5])));
                        }
                    }
                }
                //Set the Color
                Renderer renderer = gameObjects[i].GetComponentsInChildren<Transform>()[1].gameObject.GetComponent<Renderer>();
                if (i == 0)
                {
                    renderer.material.SetColor("_Color", Color.white);
                }
                else if (words[0] == "traj")
                {
                    renderer.material.SetColor("_Color", gameObjects[int.Parse(words[1])].GetComponent<Renderer>().material.GetColor("_Color"));
                }
                else if (words.Length > 6)
                {
                    renderer.material.SetColor("_Color", gradient.Evaluate(float.Parse(words[6])));
                }
                else
                {
                    renderer.material.SetColor("_Color", gradient.Evaluate(float.Parse(words[3])));
                }
            }
            Debug.Log("Initialized Objects");
            initialized = 2;
        }
    }

    //Get Request and change position
    IEnumerator GetRequestUpdate()
    {
        //Submit Web Request and log incoming errors and data
        UnityWebRequest r = UnityWebRequest.Get(url_update);
        yield return r.SendWebRequest();
        if (r.isNetworkError || r.isHttpError) // If you are not getting the server requests, reset the initialization
        {
            failCount = failCount + 1;
            if (failCount > 5 | r.downloadHandler.text.Equals("Done"))
            {
                initialUrlGot = false;
                trajectories = false;
                task1 = false;
                initialized = 1;
                deinitialize();
            }
            yield break;
        }
        else
        {
            failCount = 0;
            if (r.downloadHandler.text.Equals("Done"))
            {
                initialUrlGot = false;
                trajectories = false;
                task1 = false;
                initialized = 1;
                deinitialize();
                yield break;
            }
            
            //Parse data and alter objects
            string[] lines = r.downloadHandler.text.Split('\n');
            r.Dispose();
            // Display the two highest belief trajectories
            float highest_traj = 0.0f;
            float mid_traj = 0.0f;
            int[] traj_nums_temp = { 0, 0 };
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                string[] words = line.Split('\t');


                //Alter the object
                if (i == 0)
                {   
                    gameObjects[1].transform.position = coordTransform(new Vector3(float.Parse(words[0]), float.Parse(words[1]), float.Parse(words[2])));
                    gameObjects[2].transform.position = coordTransform(new Vector3(float.Parse(words[3]), float.Parse(words[4]), float.Parse(words[5])));
                    gameObjects[3].transform.position = coordTransform(new Vector3(float.Parse(words[6]), float.Parse(words[7]), float.Parse(words[8])));
                    gameObjects[4].transform.position = coordTransform(new Vector3(float.Parse(words[9]), float.Parse(words[10]), float.Parse(words[11])));
                    gameObjects[5].transform.position = coordTransform(new Vector3(float.Parse(words[12]), float.Parse(words[13]), float.Parse(words[14])));
                    gameObjects[6].transform.position = coordTransform(new Vector3(float.Parse(words[15]), float.Parse(words[16]), float.Parse(words[17])));
                    gameObjects[7].transform.position = coordTransform(new Vector3(float.Parse(words[18]), float.Parse(words[19]), float.Parse(words[20])));
                    gameObjects[8].transform.position = coordTransform(new Vector3(float.Parse(words[21]), float.Parse(words[22]), float.Parse(words[23])));
                }
                else
                {
                    float confidence = float.Parse(words[0]);
                    Renderer renderer = gameObjects[i].GetComponentsInChildren<Transform>()[1].gameObject.GetComponent<Renderer>();
                    if (confidence < latch_point)
                    {
                        renderer.material.SetColor("_Color", gradient.Evaluate(float.Parse(words[0])));
                        if (trajectories)
                        {
                            Transform[] children = gameObjects[i].GetComponentsInChildren<Transform>();
                            if (i == traj_nums[0] | i == traj_nums[1])
                            {
                                for (int j = 1; j < children.Length; j++)
                                {
                                    children[j].gameObject.GetComponent<Renderer>().material.SetColor("_Color", gradient.Evaluate(confidence));
                                }
                            }
                            else
                            {
                                for (int j = 1; j < children.Length; j++)
                                {
                                    children[j].gameObject.GetComponent<Renderer>().material.SetColor("_Color", new Color(0.0f, 0.0f, 0.0f, 0.0f));
                                }
                            }
                            if (confidence > highest_traj)
                            {
                                mid_traj = highest_traj;
                                highest_traj = float.Parse(words[0]);
                                traj_nums_temp[1] = traj_nums_temp[0];
                                traj_nums_temp[0] = i;
                            }
                            else if (confidence > mid_traj)
                            {
                                mid_traj = float.Parse(words[0]);
                                traj_nums_temp[1] = i;
                            }
                        }
                    }
                    else
                    {
                        for (int j = 1; j < gameObjects.Length; j++)
                        {
                            gameObjects[j].GetComponent<Renderer>().material.SetColor("_Color", new Color(0.0f, 0.0f, 0.0f, 0.0f));
                        }
                        renderer.material.SetColor("_Color", new Color(1.0f, 0.5f, 0.0f, 1.0f));
                        if (trajectories)
                        {
                            Transform[] children = gameObjects[i].GetComponentsInChildren<Transform>();
                            for (int j = 0; j < children.Length; j++)
                            {
                                if (j != 0)
                                {
                                    children[j].gameObject.GetComponent<Renderer>().material.SetColor("_Color", new Color(1.0f, 0.5f, 0.0f, 1.0f));
                                }
                            }
                        }
                        break;
                    }
                }
            }
            traj_nums = traj_nums_temp;
        }
    }
    //Coordinate transform between two points
    // @param worldpoint: Vector3 that represents point given from robot
    //@return Vector3 Vector3 that represents Unity world
    public Vector3 coordTransform(Vector3 worldPoint)
    {
        worldPoint.x = worldPoint.x - 0.1f;
        worldPoint.z = worldPoint.z - 0.05f;
        // worldPoint.y = -worldPoint.y; //Convert back to Unity LH coordinate system

        Vector3 unityPoint = invTransMatrix.MultiplyPoint(worldPoint);


        return unityPoint;
    }

    //Handles Rotation for robot arm
    public Quaternion getArmRotation(Vector3 eulerAngles)
    {
        //1. Calculate theta net
        //2. Create rotation matrix: R should rotate the end effector in relation to robot frame
        //3. Zero out InvTransMatrix translational elements
        //4. InvTransMatrix * R: Should mean to apply robot -> Unity on the given rotation
        //5.Compute Euler angles of that, apply to unity object
        Matrix4x4 rX = new Matrix4x4();
        Matrix4x4 rY = new Matrix4x4();
        Matrix4x4 rZ = new Matrix4x4();
        float thetaNetX = eulerAngles.x;
        float thetaNetY = (eulerAngles.y - (float)Math.PI / 4);
        float thetaNetZ = eulerAngles.z;

        //RX Begin
        //first row
        rX.m00 = 1;
        rX.m01 = 0;
        rX.m02 = 0;
        rX.m03 = 0;
        //second row
        rX.m10 = 0;
        rX.m11 = (float)Math.Cos(thetaNetX);
        rX.m12 = -(float)Math.Sin(thetaNetX);
        rX.m13 = 0;
        //third row
        rX.m20 = 0;
        rX.m21 = (float)Math.Sin(thetaNetX);
        rX.m22 = (float)Math.Cos(thetaNetX);
        rX.m23 = 0;
        //fourth row
        rX.m30 = 0;
        rX.m31 = 0;
        rX.m32 = 0;
        rX.m33 = 1;

        //RY
        rY.m00 = (float)Math.Cos(thetaNetY);
        rY.m01 = 0;
        rY.m02 = (float)Math.Sin(thetaNetY);
        rY.m03 = 0;
        //second row
        rY.m10 = 0;
        rY.m11 = 1;
        rY.m12 = 0;
        rY.m13 = 0;
        //third row
        rY.m20 = -(float)Math.Sin(thetaNetY);
        rY.m21 = 0;
        rY.m22 = (float)Math.Cos(thetaNetY);
        rY.m23 = 0;
        //fourth row
        rY.m30 = 0;
        rY.m31 = 0;
        rY.m32 = 0;
        rY.m33 = 1;

        //RZ
        rZ.m00 = (float)Math.Cos(thetaNetZ);
        rZ.m01 = -(float)Math.Sin(thetaNetZ);
        rZ.m02 = 0;
        rZ.m03 = 0;
        //second row
        rZ.m10 = (float)Math.Sin(thetaNetZ);
        rZ.m11 = (float)Math.Cos(thetaNetZ);
        rZ.m12 = 0;
        rZ.m13 = 0;
        //third row
        rZ.m20 = 0;
        rZ.m21 = 0;
        rZ.m22 = 1;
        rZ.m23 = 0;
        //fourth row
        rZ.m30 = 0;
        rZ.m31 = 0;
        rZ.m32 = 0;
        rZ.m33 = 1;

        Matrix4x4 r = rZ * rY * rX;

        //take invTransmatrix, take away translational compent 
        Matrix4x4 noTranslationInvMatrix = invTransMatrix;
        noTranslationInvMatrix.m03 = 0;
        noTranslationInvMatrix.m13 = 0;
        noTranslationInvMatrix.m23 = 0;
        Quaternion unityRotation = (noTranslationInvMatrix * r).rotation;
        //Debug.Log("Unity Rotation: " + unityRotation.ToString());
        return unityRotation;

    }

    public void deinitialize()
    {
        if (gameObjects == null)
        {
            return;
        }
        foreach (GameObject o in gameObjects)
        {
            Destroy(o.gameObject);
        }
        gameObjects = null;
    }

}
