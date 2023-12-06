using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameObjectColor : MonoBehaviour
{
    public void UpdateObjectColor(string colortype)
    {
        // Get the Renderer component from sphere
        GameObject sphere = GameObject.Find("Sphere");
        //Renderer sphereRenderer = Commander.Sphere.GetComponent<Renderer>();
        Renderer sphereRenderer = sphere.GetComponent<Renderer>();

        if (colortype == "red")
            sphereRenderer.material.SetColor("_Color", Color.red);
        else if (colortype == "blue")
            sphereRenderer.material.SetColor("_Color", Color.blue);
        else if (colortype == "green")
            sphereRenderer.material.SetColor("_Color", Color.green);



    }
}
