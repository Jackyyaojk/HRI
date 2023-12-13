using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.WSA;
using UnityEngine.XR.WSA.Input;
using UnityEngine.UI;
using TMPro;

public class InitScript : MonoBehaviour
{
    /// <summary>
    /// The sphere prefab.
    /// </summary>
    public GameObject spherePrefab;

    //Text Prefab
    public TextMeshPro guiText;


    /// <summary>
    /// Use the recognizer to detect air taps.
    /// </summary>
    private GestureRecognizer recognizer;

    /// <summary>
    /// True if we are 1) creating + saving an anchor or 2) looking for an anchor.
    /// </summary>
    protected bool tapExecuted = false;


    // Counter & Max number of spheres any gestures after intialization won't get placed
    public int objectCounter = 0;
    public static int maxObjects = 3;
    //List of Vector3 to store object key locations
    public Vector3[] objectLocations = new Vector3[4];
    public GameObject[] spheres = new GameObject[4];
    /// <summary>
    /// The sphere rendered to show the position of the CloudSpatialAnchor.
    /// </summary>




    // Start is called before the first frame update
    void Start()
    {
        recognizer = new GestureRecognizer();

        recognizer.StartCapturingGestures();

        recognizer.SetRecognizableGestures(GestureSettings.Tap);

        recognizer.Tapped += HandleTap;
    }


    // Update is called once per frame
    void Update()
    {

    }
    /// <summary>
    /// Called by GestureRecognizer when a tap is detected.
    /// </summary>
    /// <param name="tapEvent">The tap.</param>    
    public void HandleTap(TappedEventArgs tapEvent)
    {
        if (tapExecuted)
        {
            return;
        }


        // Construct a Ray using forward direction of the HoloLens.
        Ray GazeRay = new Ray(tapEvent.headPose.position, tapEvent.headPose.forward);

        // Raycast to get the hit point in the real world.
        RaycastHit hitInfo;
        Physics.Raycast(GazeRay, out hitInfo, float.MaxValue);
        if ((objectCounter <= maxObjects) && !(hitInfo.point.Equals(new Vector3(0.0f, 0.0f, 0.0f))))
        {
            
            this.CreateAndSaveSphere(hitInfo.point);
            // save the location of the hit to some list of hitinfos
            objectLocations[objectCounter] = hitInfo.point;
            objectCounter++;
            guiText.SetText(this.FontGuiHandler(objectCounter));
            

        if (guiText.Equals("")){
                objectCounter = 4;
            }
                
            
            if (objectCounter == 4)
            {
                Destroy(guiText);
                recognizer.StopCapturingGestures();

            }
        }

    }

    /// <summary>
    /// Creates a sphere at the hit point.
    /// </summary>
    /// <param name="hitPoint">The hit point.</param>
    protected virtual void CreateAndSaveSphere(Vector3 hitPoint)
    {
        // Create a white sphere.
        Debug.Log("HITPOINT!!!!!");
        Debug.Log(hitPoint);
        spheres[objectCounter] = Instantiate(spherePrefab, hitPoint, Quaternion.identity);
        Material sphereMaterial = spheres[objectCounter].GetComponent<MeshRenderer>().material;
        sphereMaterial.color = Color.green;
        Debug.Log("ASA Info: Created a local anchor.");
    }

    public void destroySpheres()
    {
        if (spheres == null)
        {
            return;
        }
        foreach (GameObject o in spheres)
        {
            Destroy(o.gameObject);
        }
        spheres = null;
    }

    //Changes Text based on which object is selected
    public string FontGuiHandler(int numObject)
    {
        if (objectCounter == 0)
        {
            return "Select Robot Base";
        }
        else if (objectCounter == 1)
        {
            return "Select Robot X axis";
        }
        else if (objectCounter == 2)
        {
            return "Select Robot Y Axis";
        }
        else if (objectCounter == 3)
        {
            return "Select Robot Z axis";
        }
        else
            objectCounter = 4;
        return objectCounter.ToString();
    }

    //Calculate Transformation Matrix 
    //Old Method for Transformation matrix, not using now
    public Matrix4x4 calculateTransMatrix(Vector3[] locationList)
    {
      

        float[] xVals = new float[4];
        float[] yVals = new float[4];
        float[] zVals = new float[4];
        //Transform Vector3 into floats
        //Indcies for lists ie xVals[0] = X Value of origin
        //0 = origin
        //1 = X Cords
        //2 = Y Coords
        //3 = Z Coords
        for (int i = 0; i <= 3; i++)
        {
            xVals[i] = locationList[i].x; // correcting for LH coordinate system
            yVals[i] = locationList[i].y;
            zVals[i] = locationList[i].z;
        }
        //Subtract translation
        
        Matrix4x4 transMatrix = new Matrix4x4();

        //First Row
        transMatrix.m00 = (2.5f)*(xVals[1] - xVals[0]);
        transMatrix.m01 = (2.5f)*(yVals[1] - yVals[0]);
        transMatrix.m02 = (2.5f) * (zVals[1] - zVals[0]);
        transMatrix.m03 = 0;
        //Second Row
        transMatrix.m10 = (2.5f) * (xVals[2] - xVals[0]);
        transMatrix.m11 = (2.5f) * (yVals[2] - yVals[0]);
        transMatrix.m12 = (2.5f) * (zVals[2] - zVals[0]);
        transMatrix.m13 = 0;
        //Third Row
        transMatrix.m20 = (2.5f) * (xVals[3] - xVals[0]);
        transMatrix.m21 = (2.5f) * (yVals[3] - yVals[0]);
        transMatrix.m22 = (2.5f) * (zVals[3] - zVals[0]);
        transMatrix.m23 = 0;
        //Fourth Row
        transMatrix.m30 = 0;
        transMatrix.m31 = 0;
        transMatrix.m32 = 0;
        transMatrix.m33 = 1;
      
        Vector3 b = new Vector3(xVals[0], yVals[0], zVals[0]);
        Vector3 bInv = -transMatrix.MultiplyPoint(b);
        //Add in inverted points to the matrix
        transMatrix.m03 = bInv.x;
        transMatrix.m13 = bInv.y;
        transMatrix.m23 = bInv.z;
        return transMatrix;
    }

    public float normalizeRotation(float val)
    {        
        if(val < -.3)
        {
            return -1;
        }

        else if(val < .3)
        {
            return 0;
        }
        else 
        {
            return 1;
        }
    }

    public Matrix4x4 getTransformationMatrix(Vector3[] vectors)
    {
        
        Vector3[] axes = new Vector3[3];
        // create axes vectors
        for (int i= 0; i<3; i++)
        {
            axes[i] = vectors[i+1] - vectors[0];
        }
        Matrix4x4 rotationMatrix = new Matrix4x4();

        //first row // right = x axis
        rotationMatrix.m00 = Mathf.Cos(Vector3.Angle(Vector3.right, axes[0]) * Mathf.PI / 180);
        rotationMatrix.m01 = Mathf.Cos(Vector3.Angle(Vector3.right, axes[1]) * Mathf.PI / 180);
        rotationMatrix.m02 = Mathf.Cos(Vector3.Angle( Vector3.right, axes[2]) * Mathf.PI / 180);
        rotationMatrix.m03 = vectors[0].x;
        //second row 
        rotationMatrix.m10 = Mathf.Cos(Vector3.Angle(Vector3.up, axes[0]) * Mathf.PI / 180); //up = y axis
        rotationMatrix.m11 = Mathf.Cos(Vector3.Angle(Vector3.up, axes[1]) * Mathf.PI / 180);
        rotationMatrix.m12 = Mathf.Cos(Vector3.Angle(Vector3.up, axes[2]) * Mathf.PI / 180); ;
        rotationMatrix.m13 = vectors[0].y; //y stays the same
        //third row
        rotationMatrix.m20 = Mathf.Cos(Vector3.Angle(Vector3.forward, axes[0]) * Mathf.PI / 180); //forward = z
        rotationMatrix.m21 = Mathf.Cos(Vector3.Angle(Vector3.forward, axes[1]) * Mathf.PI / 180);
        rotationMatrix.m22 = Mathf.Cos(Vector3.Angle(Vector3.forward, axes[2]) * Mathf.PI / 180);
        rotationMatrix.m23 = vectors[0].z; // z stays the same
        //forth row
        rotationMatrix.m30 = 0;
        rotationMatrix.m31 = 0;
        rotationMatrix.m32 = 0;
        rotationMatrix.m33 = 1;

        return rotationMatrix;
    }
}

