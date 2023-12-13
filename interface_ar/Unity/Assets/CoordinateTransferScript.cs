using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoordinateTransferScript : MonoBehaviour
{
    // Start is called before the first frame update
    Vector3[] locations;
    Matrix4x4 transMatrix;
    Matrix4x4 invTransMatrix;

    void Start()
    {
        locations =  GameObject.Find("InitScript").GetComponent<InitScript>().objectLocations;
        transMatrix = GameObject.Find("InitScript").GetComponent<InitScript>().calculateTransMatrix(locations);
        invTransMatrix = transMatrix.inverse;
    }

    // Update is called once per frame

    void Update()
    {
        
    }

    //Coordinate transform between two points
    // @param worldpoint: Vector3 that represents point given from robot
    //@return Vector3 Vector3 that represents Unity world
    public Vector3 coordTransform(Vector3 worldPoint)
    {
        Vector3 unityPoint = invTransMatrix.MultiplyPoint(worldPoint);
        return unityPoint;
    }
}
