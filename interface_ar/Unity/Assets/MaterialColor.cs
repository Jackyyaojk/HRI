using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MaterialColor : MonoBehaviour
{
    // Start is called before the first frame update
    MeshRenderer cubeMeshRenderer;
    private void UpdateColor(string colorarg)
    {
        cubeMeshRenderer = GetComponent<MeshRenderer>();
        if (colorarg == "red")
            cubeMeshRenderer.material.SetColor("_Color", Color.red);
        else if (colorarg == "blue")
            cubeMeshRenderer.material.SetColor("_Color", Color.blue);
        else if (colorarg == "green")
            cubeMeshRenderer.material.SetColor("_Color", Color.green);
        

    }


}
