"use client";
import {initializeApp,getApps,getApp,type FirebaseApp} from "firebase/app";
import {getAuth,RecaptchaVerifier,signInWithPhoneNumber,type Auth,type ConfirmationResult,onAuthStateChanged,signOut} from "firebase/auth";
const config={apiKey:process.env.NEXT_PUBLIC_FIREBASE_API_KEY||"",authDomain:process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN||"",projectId:process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID||"",appId:process.env.NEXT_PUBLIC_FIREBASE_APP_ID||""};
let app:FirebaseApp|null=null;let auth:Auth|null=null;
function getClientAuth(){if(typeof window==="undefined")return null;if(!config.apiKey||!config.authDomain||!config.projectId||!config.appId)throw new Error("firebase_config_missing");if(!app)app=getApps().length?getApp():initializeApp(config);if(!auth)auth=getAuth(app);return auth;}
export function subscribeAuth(callback:Parameters<typeof onAuthStateChanged>[1]){const a=getClientAuth();if(!a)return()=>{};return onAuthStateChanged(a,callback);}
export async function signOutUser(){const a=getClientAuth();if(a)await signOut(a);}
export function setupRecaptcha(id:string){const a=getClientAuth();if(!a)throw new Error("firebase_unavailable");const w=window as any;if(w.__gyaanRecaptcha)return w.__gyaanRecaptcha as RecaptchaVerifier;const v=new RecaptchaVerifier(a,id,{size:"invisible"});w.__gyaanRecaptcha=v;return v;}
export async function requestOtp(phone:string,id:string):Promise<ConfirmationResult>{const a=getClientAuth();if(!a)throw new Error("firebase_unavailable");return signInWithPhoneNumber(a,phone,setupRecaptcha(id));}
export async function currentIdToken(){const a=getClientAuth();return a?.currentUser?.getIdToken(true)||null;}
