using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class TextUpdater : MonoBehaviour
{
    public TextMeshProUGUI textDisplay;
    private int numbertest; //This is for testing u can see if it works with this first 

    private void UpdateText(string textarg)
    {
        textDisplay = GetComponent<TextMeshProUGUI>();
        textDisplay.text = textarg.ToString();// If this doesn't work try second option 
        textDisplay.SetText(textarg); // void SetText <T,U,V> (string text, T arg0, U arg1, V arg 1);
        // you can set text in this way object.SetText("text {T} text {U} text {V}", T, Uf, Vf) 
    }
}
