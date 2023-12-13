using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.WSA;
using UnityEngine.XR.WSA.Input;
using UnityEngine.Networking;
public class GazeCursor : MonoBehaviour
{
    private MeshRenderer meshRenderer;
    public GameObject initManager;
    // Start is called before the first frame update
    void Start()
    {
        // Grab the mesh renderer that's on the same object as this script.
        meshRenderer = this.GetComponent<MeshRenderer>();
    }

    // Update is called once per frame
    void Update()
    {
        // Do a raycast into the world based on the user's
        // head position and orientation.
        var headPosition = Camera.main.transform.position;
        var gazeDirection = Camera.main.transform.forward;
        Ray GazeRay = new Ray(headPosition, gazeDirection);
        
        RaycastHit hitInfo;
        // Display the cursor mesh.
        Physics.Raycast(GazeRay, out hitInfo, float.MaxValue);
        meshRenderer.enabled = true;
        // Move the cursor to the point where the raycast hit.
        this.transform.position = hitInfo.point;
        // Rotate the cursor to hug the surface of the hologram.
        this.transform.rotation =
            Quaternion.FromToRotation(Vector3.up, hitInfo.normal);
        if(initManager.GetComponent<InitScript>().objectCounter == 4)
        {
            meshRenderer.enabled = false;
            Object.Destroy(this);
        }
    }
}
